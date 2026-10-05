import json, time, sys
from trinets import *
log = open('../logs/enumeration.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
for r in (1, 2):
    t0 = time.time()
    nets = rooted_networks(r)
    P(f'r={r}: rooted networks (labelled internal vertices): {len(nets)}  [{time.time()-t0:.1f}s]')
    for lab in (False, True):
        cl = reduced_trinets(r, labelled=lab)
        P(f'  reduced trinets with strict level-{r} 3-blob, {"leaf-labelled" if lab else "unlabelled"} classes: {len(cl)}')
        for i, c in enumerate(cl):
            P(f'    class {i}: rootings={c["n_rootings"]}  sd_edges={c["sd"]}')
        json.dump([{'sd': c['sd'], 'rooted': c['rooted'], 'n_rootings': c['n_rootings']} for c in cl],
                  open(f'../data/trinets_level{r}_{"labelled" if lab else "unlabelled"}.json', 'w'), indent=1)
