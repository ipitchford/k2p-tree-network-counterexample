"""For each non-anchored level-3 shape: is there a reticulation edge c such that N - c (dangling parts removed,
degree-2 vertices and parallel arcs suppressed, but no 2-sub-blob suppression needed) is isomorphic to one of
the non-anchored level-2 trinets L2-0, L2-1, L2-2?  If so, the propagation lemma (implicit function theorem
at the regular, strictly interior certificate points) proves that N's K2P model meets the star-tree model."""
import json
from anchoring import cleanup, suppress_2subblobs, LEAVES
from trinets import sd_graph, is_iso, analyse_sd
lev2 = json.load(open('../data/trinets_level2_unlabelled.json'))
bad2 = {f'L2-{i}': sd_graph([list(e) for e in lev2[i]['sd']], labelled=False) for i in (0, 1, 2)}
res3 = {r['id']: r for r in json.load(open('../data/level3_analysis.json'))}
shapes = json.load(open('../data/trinets_level3_unlabelled.json'))
out = {}
for i, c in enumerate(shapes):
    r = res3[i]
    if r['anchored_at']:
        continue
    sd = [tuple(e) for e in c['sd']]
    rets = sorted({e[1] for e in sd if e[2] and sum(1 for f in sd if f[1] == e[1] and f[2]) == 2})
    route = None
    for h in rets:
        for j, e in enumerate(sd):
            if e[1] == h and e[2]:
                sub = cleanup([f for k, f in enumerate(sd) if k != j])
                if suppress_2subblobs(sub) != sub and sorted(suppress_2subblobs(sub)) != sorted(sub):
                    continue      # would need 2-sub-blob suppression: skip this route
                g = sd_graph([list(f) for f in sub], labelled=False)
                for name, g2 in bad2.items():
                    if is_iso(g, g2):
                        route = (h, e, name); break
            if route: break
        if route: break
    out[i] = dict(numerical_zero_found=r['Theta0']['interior_zeros'] > 0, propagation=route)
json.dump({k: dict(v, propagation=list(v['propagation']) if v['propagation'] else None) for k, v in out.items()},
          open('../data/level3_propagation.json', 'w'), indent=1)
n = len(out)
prop = [k for k, v in out.items() if v['propagation']]
num = [k for k, v in out.items() if v['numerical_zero_found']]
neither = [k for k, v in out.items() if not v['propagation'] and not v['numerical_zero_found']]
print(f'non-anchored level-3 shapes: {n}')
print(f'  proved bad by propagation from L2-0/L2-1/L2-2: {len(prop)}')
print(f'  numerical interior zero found: {len(num)}; of which also propagation: {len(set(prop) & set(num))}')
print(f'  neither (need intensive search): {len(neither)} -> {neither}')
from collections import Counter
print('  propagation sources:', Counter(v['propagation'][2] for v in out.values() if v['propagation']))
