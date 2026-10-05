"""Level-3 test of the anchoring conjecture: for each reduced strict level-3 trinet shape, decide anchoring
(Theorem C: anchored => distinguishable) and search numerically for interior tree-network coincidences on
Theta0 (edge region X <= (1+Y)/2) and in the embeddable region (X^2 <= Y)."""
import json, sys, numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares
from k2p_model import K2PNetwork, log_e
from anchoring import anchored, LEAVES
shapes = json.load(open('../data/trinets_level3_unlabelled.json'))
sig = lambda u: 1/(1+np.exp(-u))
def work(i):
    sd = [tuple(e) for e in shapes[i]['sd']]
    anc = [x for x in LEAVES if anchored(sd, x)]
    net = K2PNetwork(sd, f'L3-{i}')
    m = len(sd); r = len(net.rets)
    blob = [j for j in range(m) if j not in net.leaf_edges().values()]
    rng = np.random.default_rng(i)
    out = {'id': i, 'anchored_at': anc, 'n_reticulation_leaves': sum(1 for x in LEAVES if any(x in e[:2] and (e[0] in net.rets or e[1] in net.rets) for e in sd))}
    for R, cap in (('Theta0', lambda y: (1+y)/2), ('embeddable', np.sqrt)):
        best = (np.inf, 0.0); found = 0
        starts = 40 if R == 'Theta0' else 25
        for t in range(starts):
            def unpack(u):
                y = sig(u[m:2*m]); x = cap(y)*sig(u[:m]); d = sig(u[2*m:]); return x, y, d
            s = least_squares(lambda u: log_e(net.numeric(*unpack(u))), rng.normal(0, 1.5, 2*m+r), method='trf',
                              xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=1500)
            x, y, d = unpack(s.x)
            margin = float(min(np.min(y[blob]), np.min(1-y[blob]), np.min(x[blob]), np.min(cap(y[blob])-x[blob]), np.min(d), np.min(1-d)))
            res = float(np.linalg.norm(s.fun))
            if res < 1e-10 and margin > 1e-3: found += 1
            if res < best[0]: best = (res, margin)
            if found >= 2: break
        out[R] = dict(interior_zeros=found, best_residual=best[0], best_margin=best[1])
    return out
if __name__ == '__main__':
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as p:
        res = p.map(work, range(len(shapes)))
    json.dump(res, open('../data/level3_analysis.json', 'w'), indent=1)
    anc = [r for r in res if r['anchored_at']]; non = [r for r in res if not r['anchored_at']]
    print(f'level-3 shapes: {len(res)}; anchored: {len(anc)}; not anchored: {len(non)}')
    print('anchored shapes with an interior coincidence found (would contradict Theorem C):',
          [r['id'] for r in anc if r['Theta0']['interior_zeros'] > 0])
    print('non-anchored shapes WITHOUT an interior coincidence found on Theta0:',
          [(r['id'], r['Theta0']['best_residual'], r['Theta0']['best_margin']) for r in non if r['Theta0']['interior_zeros'] == 0])
    print('embeddable region: shapes with interior coincidences found:', [r['id'] for r in res if r['embeddable']['interior_zeros'] > 0])
