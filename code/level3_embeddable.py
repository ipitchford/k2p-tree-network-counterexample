"""Embeddable region at level 3: which non-anchored level-3 shapes reach L2-1 (the K_{2,3} trinet, whose
coincidence persists for embeddable, transition-biased parameters) by a single reticulation-edge deletion?
Compare with the shapes for which the numerical search found interior embeddable coincidences."""
import json
from anchoring import cleanup, suppress_2subblobs, anchored, LEAVES
from trinets import sd_graph, is_iso
lev2 = json.load(open('../data/trinets_level2_unlabelled.json'))
gK = sd_graph([list(e) for e in lev2[1]['sd']], labelled=False)
shapes = json.load(open('../data/trinets_level3_unlabelled.json'))
res3 = {r['id']: r for r in json.load(open('../data/level3_analysis.json'))}
viaK = []
for i, c in enumerate(shapes):
    sd = [tuple(e) for e in c['sd']]
    rets = sorted({e[1] for e in sd if e[2] and sum(1 for f in sd if f[1] == e[1] and f[2]) == 2})
    for j, e in enumerate(sd):
        if not (e[2] and e[1] in rets): continue
        sub = cleanup([f for k, f in enumerate(sd) if k != j])
        if sorted(suppress_2subblobs(sub)) != sorted(sub): continue
        if is_iso(sd_graph([list(f) for f in sub], labelled=False), gK):
            viaK.append(i); break
numR2 = [i for i, r in res3.items() if r['embeddable']['interior_zeros'] > 0]
print('level-3 shapes reaching the K_{2,3} trinet by one deletion:', sorted(viaK))
print('level-3 shapes with numerical interior embeddable coincidences:', sorted(numR2))
print('numerical but no K23 route:', sorted(set(numR2) - set(viaK)), '; K23 route but not found numerically:', sorted(set(viaK) - set(numR2)))
print('anchored among these?', [i for i in set(viaK) | set(numR2) if res3[i]['anchored_at']])
json.dump(dict(viaK23=sorted(viaK), numerical_embeddable=sorted(numR2)), open('../data/level3_embeddable.json', 'w'), indent=1)
