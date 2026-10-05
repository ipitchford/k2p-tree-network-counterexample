"""Fail-closed finite release checks; universal proofs remain manuscript obligations.
Usage: python code/verify_release.py --out NEW_DIRECTORY
Historical enumeration is checked by isomorphism against a separately generated
graph-atlas inventory. This command does not rerun the rooted exhaustive search.
"""
import argparse
import copy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import networkx as nx
import sympy as sp
from anchoring import LEAVES, anchored, cleanup, suppress_2subblobs
from trinets import sd_graph
from k2p_model import K2PNetwork

ROOT = Path(__file__).resolve().parents[1]

def require(ok, detail):
    if not ok:
        raise ValueError(detail)

def load(path):
    return json.loads(path.read_text())

def matcher(a, b):
    return nx.algorithms.isomorphism.DiGraphMatcher(
        sd_graph(a, False), sd_graph(b, False),
        node_match=lambda x,y: x['lab'] == y['lab'],
        edge_match=lambda x,y: x['k'] == y['k'])

def run_certificate(cert, output, region='Theta0', success=True):
    command = [sys.executable] + (['-O'] if sys.flags.optimize else [])
    command += [str(ROOT/'code/certify.py'), str(cert), '--require-region', region]
    if success:
        command += ['--output', str(output.with_suffix('.json'))]
    result = subprocess.run(command, text=True, capture_output=True)
    output.with_suffix('.log').write_text(result.stdout + result.stderr)
    require((result.returncode == 0) == success,
            f'certificate outcome incorrect: {cert.name}; exit {result.returncode}')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'finite exact checks; not formal verification', 'optimized': bool(sys.flags.optimize)}
    ranks = {}
    for name in ('L2-0', 'L2-1', 'L2-2'):
        file = ROOT/f'certificates/{name}_example.json'
        run_certificate(file, args.out/name, 'kappa_ge_1' if name == 'L2-1' else 'Theta0')
        cert = load(file)
        model = K2PNetwork(cert['sd'], name)
        substitutions = {}
        for i,(x,y) in cert['params'].items():
            substitutions[model.X[int(i)]] = sp.Rational(x)
            substitutions[model.Y[int(i)]] = sp.Rational(y)
        substitutions.update({model.d[i]: sp.Rational(cert['deltas'][h]) for i,h in enumerate(model.rets)})
        q = sp.Matrix([model.coords[k] for k in model.coord_names()])
        dq = q.jacobian(model.params).subs(substitutions)
        c = sp.symbols('a:9')
        expressions = []
        for i in range(3):
            j,k = [n for n in range(3) if n != i]
            expressions.append(c[i]**2*c[3+j]*c[3+k] - c[3+i]*c[6+i]**2)
        chain = sp.Matrix(expressions).jacobian(c).subs(dict(zip(c,q.subs(substitutions)))) * dq
        cols3 = [0,1,3]
        det3 = chain[:,cols3].det()
        cols9 = list(dq.rref()[1])
        require(len(cols9) == 9 and det3 != 0, f'{name}: rank failure')
        det9 = dq[:,cols9].det()
        require(det9 != 0, f'{name}: nine-coordinate minor failure')
        ranks[name] = {'rank_E':3, 'rank_coordinates':9,
                       'minor3_columns':[str(model.params[i]) for i in cols3],
                       'minor3':str(det3),
                       'minor9_columns':[str(model.params[i]) for i in cols9], 'minor9':str(det9)}
    report['ranks'] = ranks
    coverage = {}
    for level, expected in ((1,1),(2,6),(3,83)):
        original = load(ROOT/f'data/trinets_level{level}_unlabelled.json')
        reference = load(ROOT/f'data/atlas_level{level}.json')
        require(len(original) == len(reference) == expected, 'class count mismatch')
        used = set(); witnesses = []
        for network in reference:
            matches = []
            for i, other in enumerate(original):
                comparison = matcher(network['sd'], other['sd'])
                if comparison.is_isomorphic():
                    matches.append((i,dict(comparison.mapping)))
            require(len(matches) == 1, 'isomorphism is not unique')
            index, mapping = matches[0]
            require(index not in used, 'non-bijective class inventory')
            used.add(index); witnesses.append({'target_index':index, 'mapping':mapping})
        coverage[str(level)] = witnesses
    report['isomorphism_witnesses'] = coverage
    seeds = load(ROOT/'data/trinets_level2_unlabelled.json')[:3]
    routes = []
    for index,network in enumerate(load(ROOT/'data/trinets_level3_unlabelled.json')):
        edges = [tuple(e) for e in network['sd']]
        anchors = [leaf for leaf in LEAVES if anchored(edges,leaf)]
        found = []
        for i,edge in enumerate(edges):
            if not edge[2]:
                continue
            reduced = cleanup(edges[:i]+edges[i+1:])
            if sorted(reduced) != sorted(suppress_2subblobs(reduced)):
                continue
            for s,seed in enumerate(seeds):
                comparison = matcher(reduced, seed['sd'])
                if comparison.is_isomorphic():
                    found.append({'deleted_edge_index':i,'target':s,'mapping':dict(comparison.mapping)})
        require(bool(anchors) != bool(found), f'unclassified or contradictory shape {index}')
        routes.append({'index':index,'anchors':anchors,'routes':found})
    summary = [sum(bool(r['anchors']) for r in routes), sum(bool(r['routes']) for r in routes),
               sum(any(t['target']==1 for t in r['routes']) for r in routes)]
    require(summary == [30,53,14], 'classification totals changed')
    report['classification'] = routes
    report['classification_totals'] = summary
    # Check the path identities used in the Jensen contradiction symbolically.
    u0,u1,u2,u3,u4,u8 = sp.symbols('u0 u1 u2 u3 u4 u8', positive=True)
    equations = [u1*u3-u2*u4, u1-u2*u3*u4]
    require(sp.expand(equations[0]-u3*equations[1]-u2*u4*(u3**2-1)) == 0, 'Jensen elimination')
    report['common_ratio_elimination'] = 'u2*u4*(u3**2-1)=0; positive u2,u4 and 0<u3<1 contradict'
    base = load(ROOT/'certificates/L2-1_example.json')
    controls = []
    for kind in ('binomial', 'stochasticity', 'weight'):
        bad = copy.deepcopy(base)
        if kind == 'binomial':
            bad['params']['0'][0] = str(Fraction(bad['params']['0'][0])+Fraction(1,100000))
        elif kind == 'stochasticity':
            bad['params']['0'][0] = '2'
        else:
            bad['deltas'][next(iter(bad['deltas']))] = '1'
        path = args.out/f'negative-{kind}.json'; path.write_text(json.dumps(bad))
        run_certificate(path, args.out/f'rejected-{kind}', success=False)
        controls.append(kind)
    run_certificate(ROOT/'certificates/L2-0_example.json', args.out/'rejected-embeddability', 'embeddable', False)
    report['negative_controls_rejected'] = controls + ['embeddability']
    report['passed'] = True
    (args.out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'seeds':3,'probabilities_per_seed':64,'classes':[1,6,83],
                      'classification':summary,'negative_controls':4,'optimized':bool(sys.flags.optimize)}))

if __name__ == '__main__':
    main()
