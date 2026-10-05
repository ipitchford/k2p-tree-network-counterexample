"""Strict embeddable-region search for selected level-3 shapes: a coincidence counts only if |e| < 1e-13 and every
inequality (Y in (0,1), X > 0, X^2 < Y, delta in (0,1)) holds with margin > 1e-2."""
import json, sys, numpy as np
from scipy.optimize import least_squares
from k2p_model import K2PNetwork, log_e
shapes = json.load(open('../data/trinets_level3_unlabelled.json'))
sig = lambda u: 1/(1+np.exp(-u))
for i in [int(a) for a in sys.argv[1].split(',')]:
    sd = [tuple(e) for e in shapes[i]['sd']]; net = K2PNetwork(sd, f'L3-{i}'); m = len(sd); r = len(net.rets)
    blob = [j for j in range(m) if j not in net.leaf_edges().values()]
    def unpack(u):
        y = sig(u[m:2*m]); x = np.sqrt(y)*sig(u[:m]); d = sig(u[2*m:]); return x, y, d
    rng = np.random.default_rng(500 + i); best = (np.inf, 0); good = 0
    for t in range(int(sys.argv[2])):
        s = least_squares(lambda u: log_e(net.numeric(*unpack(u))), rng.normal(0, 1.2, 2*m+r), method='trf', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
        x, y, d = unpack(s.x)
        margin = float(min(np.min(y[blob]), np.min(1-y[blob]), np.min(x[blob]), np.min(y[blob]-x[blob]**2), np.min(d), np.min(1-d)))
        res = float(np.linalg.norm(s.fun))
        if res < 1e-13 and margin > 1e-2: good += 1
        if res < 1e-9 and margin > best[1]: best = (res, margin)
    print(f'L3-{i}: strict embeddable coincidences {good}/{sys.argv[2]}; best margin among |e|<1e-9: {best[1]:.3e} (|e|={best[0]:.1e})', flush=True)
