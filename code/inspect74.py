import json, numpy as np
from scipy.optimize import least_squares
from k2p_model import K2PNetwork, log_e
from anchoring import anchored, reduce_network, LEAVES
shapes = json.load(open('../data/trinets_level3_unlabelled.json'))
sd = [tuple(e) for e in shapes[74]['sd']]
print('shape 74 edges:', sd)
print('reduced form:', reduce_network(sd))
tr = []
print('anchored at:', [x for x in LEAVES if anchored(sd, x, trace=tr)], tr)
net = K2PNetwork(sd, 'L3-74'); m = len(sd); r = len(net.rets)
print('reticulations', net.rets, 'leaf edges', net.leaf_edges())
sig = lambda u: 1/(1+np.exp(-u))
rng = np.random.default_rng(74)
def unpack(u):
    y = sig(u[m:2*m]); x = np.sqrt(y)*sig(u[:m]); d = sig(u[2*m:]); return x, y, d
blob = [j for j in range(m) if j not in net.leaf_edges().values()]
for t in range(25):
    s = least_squares(lambda u: log_e(net.numeric(*unpack(u))), rng.normal(0, 1.5, 2*m+r), method='trf', xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=1500)
    x, y, d = unpack(s.x)
    margin = float(min(np.min(y[blob]), np.min(1-y[blob]), np.min(x[blob]), np.min(np.sqrt(y[blob])-x[blob]), np.min(d), np.min(1-d)))
    if np.linalg.norm(s.fun) < 1e-9:
        c = net.numeric(x, y, d); e = log_e(c)
        print(f'start {t}: |e|={np.linalg.norm(s.fun):.2e} margin={margin:.2e} e={e} d={d}')
        print('   X=', np.round(x, 4).tolist()); print('   Y=', np.round(y, 4).tolist())
