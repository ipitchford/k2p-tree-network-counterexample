"""Polish the best embeddable near-coincidence for level-3 shape 5 in 50-digit arithmetic (Newton on three
parameters, others frozen) and report the tightest constraint."""
import json, numpy as np, mpmath as mp, sympy as sp
from scipy.optimize import least_squares
from k2p_model import K2PNetwork, log_e
mp.mp.dps = 50
shapes = json.load(open('../data/trinets_level3_unlabelled.json'))
i = 5; sd = [tuple(e) for e in shapes[i]['sd']]; net = K2PNetwork(sd, f'L3-{i}'); m = len(sd); r = len(net.rets)
print('edges', sd)
blob = [j for j in range(m) if j not in net.leaf_edges().values()]
sig = lambda u: 1/(1+np.exp(-u))
def unpack(u):
    y = sig(u[m:2*m]); x = np.sqrt(y)*sig(u[:m]); d = sig(u[2*m:]); return x, y, d
rng = np.random.default_rng(505); best = None
for t in range(60):
    s = least_squares(lambda u: log_e(net.numeric(*unpack(u))), rng.normal(0, 1.2, 2*m+r), method='trf', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
    x, y, d = unpack(s.x)
    gaps = {**{f'Y{j}': y[j] for j in blob}, **{f'1-Y{j}': 1-y[j] for j in blob}, **{f'X{j}': x[j] for j in blob},
            **{f'Y{j}-X{j}^2': y[j]-x[j]**2 for j in blob}, **{f'd{j}': d[j] for j in range(r)}, **{f'1-d{j}': 1-d[j] for j in range(r)}}
    mg = min(gaps.values())
    if np.linalg.norm(s.fun) < 1e-9 and (best is None or mg > best[0]):
        best = (mg, x, y, d, sorted(gaps.items(), key=lambda kv: kv[1])[:4])
mg, x, y, d, tight = best
print('best margin', mg, 'tightest', tight)
# Newton in high precision on three blob X-parameters with full-rank Jacobian
vals = {**{net.X[j]: mp.mpf(x[j]) for j in range(m)}, **{net.Y[j]: mp.mpf(y[j]) for j in range(m)}, **{net.d[j]: mp.mpf(d[j]) for j in range(r)}}
E = net.E
cands = [net.X[j] for j in blob]
import itertools
f_num = sp.lambdify(net.params, E, 'mpmath')
for trio in itertools.combinations(cands, 3):
    J = sp.Matrix([[sp.diff(e, p) for p in trio] for e in E])
    Jf = sp.lambdify(net.params, J, 'mpmath')
    args = [vals[p] for p in net.params]
    if abs(mp.det(mp.matrix(Jf(*args)))) > 1e-12:
        break
z = [vals[p] for p in trio]
for it in range(60):
    args = [vals[p] if p not in trio else z[trio.index(p)] for p in net.params]
    F = mp.matrix(f_num(*args)); Jm = mp.matrix(Jf(*args))
    z = [z[k] - (mp.lu_solve(Jm, F))[k] for k in range(3)]
args = [vals[p] if p not in trio else z[trio.index(p)] for p in net.params]
print('Newton on', trio, ': |E| after polishing =', mp.nstr(mp.norm(mp.matrix(f_num(*args))), 5))
print('polished values', [mp.nstr(v, 12) for v in z], 'vs start', [mp.nstr(vals[p], 12) for p in trio])
for p, v in zip(trio, z):
    j = int(str(p)[1:]); print(f'  {p}: X={mp.nstr(v,8)}, Y={y[j]:.6f}, Y-X^2={mp.nstr(mp.mpf(y[j])-v**2,6)}')
