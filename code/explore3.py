"""Search for K2P tree-network intersections under three nested edge-parameter regions:
 R1 = Theta_0 (stochastic, positive definite): 0<Y<1, 0<X<=(1+Y)/2
 R2 = embeddable (continuous-time K2P):        0<Y<1, 0<X<=sqrt(Y)
 R3 = embeddable with transition rate >= transversion rate: 0<Y<1, 0<X<=Y
Tree parameters induced at a zero are checked against the same region."""
import json, numpy as np
from scipy.optimize import least_squares
from k2p_model import *
log = open('../logs/explore3.log','w')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); log.write(s+'\n'); log.flush()
rng = np.random.default_rng(11)
sig = lambda u: 1/(1+np.exp(-u))
cap = {'R1': lambda y:(1+y)/2, 'R2': lambda y:np.sqrt(y), 'R3': lambda y:y}
nets = [K2PNetwork(c['sd'], f'L2-{i}') for i,c in enumerate(json.load(open('../data/trinets_level2_unlabelled.json')))]
found = {}
for R in ('R1','R2','R3'):
    for net in nets:
        m=len(net.sd); r=len(net.rets)
        def unpack(u):
            y=sig(u[m:2*m]); x=cap[R](y)*sig(u[:m]); d=sig(u[2*m:2*m+r]); return x,y,d
        def res(u):
            return log_e(net.numeric(*unpack(u)))
        sols=[]
        for t in range(60):
            s=least_squares(res, rng.normal(0,1.5,2*m+r), method='trf', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
            if np.linalg.norm(s.fun)<1e-12 and np.max(np.abs(s.x))<8:
                x,y,d=unpack(s.x); c=net.numeric(x,y,d)
                A=[c['A1'],c['A2'],c['A3']]; B=[c['B1'],c['B2'],c['B3']]
                tx=np.array([np.sqrt(A[1]*A[2]/A[0]),np.sqrt(A[0]*A[2]/A[1]),np.sqrt(A[0]*A[1]/A[2])])
                ty=np.array([np.sqrt(B[1]*B[2]/B[0]),np.sqrt(B[0]*B[2]/B[1]),np.sqrt(B[0]*B[1]/B[2])])
                ok = np.all(ty<1) and np.all(tx>0) and np.all(tx<=cap[R](ty))
                sols.append((np.max(np.abs(s.x)), ok, x, y, d, tx, ty))
        nok=sum(1 for s in sols if s[1])
        P(f'{R} {net.name}: interior zeros found {len(sols)}/60, with admissible induced tree {nok}')
        if nok:
            b=min([s for s in sols if s[1]], key=lambda s:s[0])
            found[f'{R}_{net.name}']=dict(X=b[2].tolist(),Y=b[3].tolist(),delta=b[4].tolist(),tree_x=b[5].tolist(),tree_y=b[6].tolist(),max_abs_u=b[0])
json.dump(found, open('../data/zeros_by_region.json','w'), indent=1)
