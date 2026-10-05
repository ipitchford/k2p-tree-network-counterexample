"""Level-4 test of the structural route behind the classification: generate reduced strict level-4 trinets by
inserting one reticulation arc into (rooted representatives of) the 83 level-3 shapes, then for each
non-anchored one look for a reticulation edge c with N - c reduced (no 2-sub-blob suppression), strict
level-3, non-anchored -- i.e. one of the 53 bad level-3 shapes."""
import json, itertools
import networkx as nx
from trinets import semidirected, analyse_sd, sd_graph, is_iso, LEAVES
from anchoring import anchored, cleanup, suppress_2subblobs
shapes3 = json.load(open('../data/trinets_level3_unlabelled.json'))
classes = {}
def add_class(sd):
    g = sd_graph([list(e) for e in sd], labelled=False)
    h = nx.weisfeiler_lehman_graph_hash(g, node_attr='lab', edge_attr='k')
    b = classes.setdefault(h, [])
    for c in b:
        if is_iso(c['g'], g): return False
    b.append({'g': g, 'sd': sd}); return True
for c in shapes3:
    par = c['rooted']['parents']
    edges = [(p, ch) for ch, ps in par.items() for p in ps]
    D = nx.DiGraph(edges)
    for (a, b), (a2, b2) in itertools.permutations(edges, 2):
        # s on (a->b), v on (a2->b2), arc s->v ; acyclic iff v is not an ancestor of s, i.e. b2 not ancestor-or-equal of a
        if b2 == a or b2 in nx.ancestors(D, a):
            continue
        newp = {k: list(v) for k, v in par.items()}
        newp['s9'] = [a]; newp[b] = [x if not (x == a and (a, b) == (a, b)) else x for x in newp[b]]
        newp[b] = ['s9' if x == a else x for x in newp[b]] if newp[b].count(a) == 1 else None
        if newp[b] is None: continue
        newp['h9'] = [a2, 's9']
        if newp[b2].count(a2) != 1: continue
        newp[b2] = ['h9' if x == a2 else x for x in newp[b2]]
        sd = semidirected(newp)
        if sd is None: continue
        info = analyse_sd(sd)
        if info and not info.get('tree') and info['ret'] == 4:
            add_class([tuple(e) for e in sd])
lev4 = [c['sd'] for b in classes.values() for c in b]
print('generated reduced strict level-4 trinet shapes:', len(lev4))
n_anch = n_route = 0; fails = []
for sd in lev4:
    if any(anchored(sd, x) for x in LEAVES):
        n_anch += 1; continue
    rets = sorted({e[1] for e in sd if e[2] and sum(1 for f in sd if f[1] == e[1] and f[2]) == 2})
    ok = False
    for j, e in enumerate(sd):
        if not (e[2] and e[1] in rets): continue
        sub = cleanup([f for k, f in enumerate(sd) if k != j])
        if sorted(suppress_2subblobs(sub)) != sorted(sub): continue
        info = analyse_sd([list(f) for f in sub])
        if info and not info.get('tree') and info['ret'] == 3 and not any(anchored(sub, x) for x in LEAVES):
            ok = True; break
    if ok: n_route += 1
    else: fails.append(sd)
print(f'anchored: {n_anch}; non-anchored with a single-deletion route to a non-anchored reduced level-3 trinet: {n_route}; without: {len(fails)}')
json.dump(dict(n=len(lev4), anchored=n_anch, route=n_route, fails=fails), open('../data/level4_sample.json', 'w'), indent=1)
