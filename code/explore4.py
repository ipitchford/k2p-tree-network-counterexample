"""Sign patterns of e = (e1,e2,e3) for level-2 trinets when edge parameters are restricted to
R2 (embeddable: X^2 <= Y) or R3 (X <= Y)."""
import json, numpy as np, collections
from k2p_model import *
log=open('../logs/explore4.log','w')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); log.write(s+'\n'); log.flush()
rng=np.random.default_rng(99)
nets=[K2PNetwork(c['sd'],f'L2-{i}') for i,c in enumerate(json.load(open('../data/trinets_level2_unlabelled.json')))]
N=400000
for R,capf in (('R2',np.sqrt),('R3',lambda y:y)):
    for net in nets:
        m=len(net.sd); r=len(net.rets)
        y=rng.uniform(0,1,(m,N)); x=capf(y)*rng.uniform(0,1,(m,N)); d=rng.uniform(0,1,(r,N))
        e=log_e(net.numeric(x,y,d))
        pat=collections.Counter(tuple('+' if v>0 else '-' for v in col) for col in e.T)
        P(f'{R} {net.name}: frac e_i<0 = {[round(float(np.mean(e[i]<0)),4) for i in range(3)]}; min_i-max: {e.max(axis=0).min():+.3e}; patterns {dict(pat)}')
