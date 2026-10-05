"""Intensive search for tree-network intersections of L2-0 and L2-2 restricted to embeddable K2P edges
(X^2 <= Y).  Records the smallest residual ||e|| found and how close the minimiser is to the boundary."""
import json, sys, numpy as np
from scipy.optimize import least_squares
from k2p_model import *
idx = int(sys.argv[1]); starts = int(sys.argv[2])
rng = np.random.default_rng(1000 + idx)
net = K2PNetwork(json.load(open('../data/trinets_level2_unlabelled.json'))[idx]['sd'], f'L2-{idx}')
m = len(net.sd); r = len(net.rets); sig = lambda u: 1/(1+np.exp(-u))
def unpack(u):
    y = sig(u[m:2*m]); x = np.sqrt(y)*sig(u[:m]); d = sig(u[2*m:]); return x, y, d
res = []
for t in range(starts):
    s = least_squares(lambda u: log_e(net.numeric(*unpack(u))), rng.normal(0, 2, 2*m+r), method='trf',
                      xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
    x, y, d = unpack(s.x)
    blob = [i for i in range(m) if i not in net.leaf_edges().values()]
    margin = min(np.min(y[blob]), np.min(1-y[blob]), np.min(x[blob]), np.min(np.sqrt(y[blob])-x[blob]), np.min(d), np.min(1-d))
    res.append((float(np.linalg.norm(s.fun)), float(margin)))
res.sort()
print(f'L2-{idx} R2: {starts} starts; 10 smallest residuals with distance of minimiser to the boundary of the region:')
for a, b in res[:10]: print(f'   |e| = {a:.3e}   boundary distance = {b:.3e}')
print('   count |e|<1e-10 with boundary distance > 1e-3:', sum(1 for a, b in res if a < 1e-10 and b > 1e-3))
