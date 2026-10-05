import json, numpy as np, sympy as sp, itertools, collections
from k2p_model import *
log = open('../logs/explore1.log','w')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); log.write(s+'\n'); log.flush()
rng = np.random.default_rng(20261005)
star = K2PNetwork([('c','1',False),('c','2',False),('c','3',False)], 'star')
P('star: q_AAA =', star.coords['AAA'], ' E =', [sp.simplify(e) for e in star.E])
nets = []
for r in (1,2):
    for i,c in enumerate(json.load(open(f'../data/trinets_level{r}_unlabelled.json'))):
        nets.append(K2PNetwork(c['sd'], f'L{r}-{i}'))
N = 200000
for net in nets:
    assert sp.simplify(net.coords['AAA'] - 1) == 0
    m = len(net.sd); r = len(net.rets)
    retleaves = sorted({e[1] for e in net.sd if e[0] in net.rets and e[1] in LEAVES} | {e[0] for e in net.sd if e[1] in net.rets and e[0] in LEAVES})
    # leaves adjacent to a reticulation vertex
    retleaves = sorted(l for l in LEAVES if any((l in e[:2]) and (e[0] in net.rets or e[1] in net.rets) for e in net.sd))
    x,y = sample_k2p_edges(rng, m, N)
    d = rng.uniform(0,1,size=(r,N))
    c = net.numeric(x,y,d)
    e = log_e(c)
    P(f'\n{net.name}: edges={net.sd}\n  reticulations={net.rets}, leaves adjacent to a reticulation={retleaves}')
    for i in range(3):
        P(f'  e{i+1}: min {e[i].min():+.4e}  max {e[i].max():+.4e}  frac<0 {np.mean(e[i]<0):.4f}')
    P(f'  min over samples of max_i e_i: {e.max(axis=0).min():+.4e}')
    P(f'  min over samples of sum_i e_i: {e.sum(axis=0).min():+.4e}')
    P(f'  min over samples of ||e||: {np.linalg.norm(e,axis=0).min():.4e}')
    pat = collections.Counter(tuple('+' if v>0 else '-' for v in col) for col in e.T)
    P('  sign patterns (e1,e2,e3):', dict(pat))
