"""JC consistency check and least-squares search for K2P tree-network intersections."""
import json, numpy as np, sympy as sp
from scipy.optimize import least_squares
from k2p_model import *
log = open('../logs/explore2.log','w')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); log.write(s+'\n'); log.flush()
rng = np.random.default_rng(7)
nets = []
for r in (1,2):
    for i,c in enumerate(json.load(open(f'../data/trinets_level{r}_unlabelled.json'))):
        nets.append(K2PNetwork(c['sd'], f'L{r}-{i}'))
# --- JC check: X=Y on every edge -> Q_JC = A1A2A3 - C^2 must be > 0 (Lemma lem:trinetJC_levk)
for net in nets:
    m=len(net.sd); r=len(net.rets); N=100000
    x=rng.uniform(0,1,(m,N)); d=rng.uniform(0,1,(r,N))
    c=net.numeric(x,x,d)
    assert np.allclose(c['C1'],c['C2']) and np.allclose(c['C1'],c['C3'])
    qjc = c['A1']*c['A2']*c['A3'] - c['C1']**2
    P(f'{net.name}: JC check min (A1A2A3-C^2)/(A1A2A3) = {np.min(qjc/(c["A1"]*c["A2"]*c["A3"])):.3e} (must be >0)')
sig = lambda u: 1/(1+np.exp(-u))
def unpack(net,u):
    m=len(net.sd); r=len(net.rets)
    y=sig(u[m:2*m]); x=(1+y)/2*sig(u[:m]); d=sig(u[2*m:2*m+r])
    return x,y,d
def resid(net,u):
    x,y,d=unpack(net,u)
    c=net.numeric(x,y,d)
    return log_e(c)
results={}
for net in nets[1:]:
    m=len(net.sd); r=len(net.rets); best=[]
    for trial in range(120):
        u0=rng.normal(0,1.5,2*m+r)
        sol=least_squares(lambda u: resid(net,u), u0, method='trf', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
        nrm=np.linalg.norm(sol.fun)
        best.append((nrm, np.max(np.abs(sol.x)), sol.x))
    best.sort(key=lambda t:t[0])
    good=[b for b in best if b[0]<1e-12]
    interior=[b for b in good if b[1]<8]
    P(f'{net.name}: residual quantiles {[f"{q:.2e}" for q in np.quantile([b[0] for b in best],[0,0.1,0.5])]}; '
      f'#|e|<1e-12: {len(good)}; of which max|u|<8 (well inside Theta0): {len(interior)}')
    if interior:
        b=min(interior,key=lambda t:t[1])
        x,y,d=unpack(net,b[2])
        P('   example interior zero: max|u|=%.2f'%b[1])
        P('   X=',np.round(x,4).tolist()); P('   Y=',np.round(y,4).tolist()); P('   delta=',np.round(d,4).tolist())
        c=net.numeric(x,y,d)
        A=[c['A1'],c['A2'],c['A3']]; B=[c['B1'],c['B2'],c['B3']]
        tx=[np.sqrt(A[1]*A[2]/A[0]),np.sqrt(A[0]*A[2]/A[1]),np.sqrt(A[0]*A[1]/A[2])]
        ty=[np.sqrt(B[1]*B[2]/B[0]),np.sqrt(B[0]*B[2]/B[1]),np.sqrt(B[0]*B[1]/B[2])]
        P('   induced star-tree params x=',np.round(tx,5).tolist(),' y=',np.round(ty,5).tolist(),
          ' admissible (0<y<1, 0<x<=(1+y)/2):', all(0<ty[i]<1 and 0<tx[i]<=(1+ty[i])/2 for i in range(3)))
        results[net.name]=dict(x=x.tolist(),y=y.tolist(),d=d.tolist(),tx=tx,ty=ty)
json.dump(results,open('../data/lsq_zeros.json','w'),indent=1)
