"""Exploratory NETWORK-ONLY shared-ratio search, not a common-ratio tree test.
The comparison tree has edge-specific K2P parameters and is now checked for
strict stochastic and continuous-time admissibility. Numerical hits are not certificates.
Both models sharing the same prescribed kappa>1 is excluded for L2-1 by Proposition 8.
"""
import json, numpy as np
from scipy.optimize import least_squares
from k2p_model import K2PNetwork, log_e
nets = json.load(open('../data/trinets_level2_unlabelled.json'))
sig = lambda u: 1/(1+np.exp(-u))
for idx in (1, 0, 2):
    net = K2PNetwork(nets[idx]['sd'], f'L2-{idx}'); m = len(net.sd); r = len(net.rets)
    blob = [j for j in range(m) if j not in net.leaf_edges().values()]
    for kappa in (2.0, 3.0, 5.0, 10.0, 30.0):
        rng = np.random.default_rng(int(kappa*10) + idx); found = 0; best = (np.inf, 0)
        for t in range(40):
            def unpack(u):
                y = sig(u[:m]); x = y ** ((kappa + 1) / 2); d = sig(u[m:]); return x, y, d
            s = least_squares(lambda u: log_e(net.numeric(*unpack(u))), rng.normal(0, 1.5, m + r), method='trf', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
            x, y, d = unpack(s.x)
            margin = float(min(np.min(y[blob]), np.min(1 - y[blob]), np.min(d), np.min(1 - d)))
            res = float(np.linalg.norm(s.fun))
            coords = net.numeric(x, y, d)
            A = np.asarray([coords[f'A{i}'] for i in (1, 2, 3)])
            B = np.asarray([coords[f'B{i}'] for i in (1, 2, 3)])
            tx = np.sqrt(np.prod(A)) / A
            ty = np.sqrt(np.prod(B)) / B
            tree_ok = bool(np.all(np.isfinite(tx)) and np.all(np.isfinite(ty))
                           and np.all(tx > 0) and np.all(ty > 0)
                           and np.all(ty < 1) and np.all(2*tx < 1+ty)
                           and np.all(tx*tx < ty))
            if res < 1e-12 and margin > 1e-2 and tree_ok: found += 1
            if res < 1e-9 and margin > best[1]: best = (res, margin)
        print(f'L2-{idx} network-only kappa={kappa:g}: numerical hits with embeddable comparison tree {found}/40; unvalidated-residual best margin: {best[1]:.3f}', flush=True)
