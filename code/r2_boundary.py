"""Which boundary do the near-zeros of L2-0 / L2-2 approach in the embeddable region?"""
import json, sys, numpy as np
from scipy.optimize import least_squares
from k2p_model import *
for idx in (0, 2):
    rng = np.random.default_rng(77 + idx)
    net = K2PNetwork(json.load(open('../data/trinets_level2_unlabelled.json'))[idx]['sd'], f'L2-{idx}')
    m = len(net.sd); r = len(net.rets); sig = lambda u: 1/(1+np.exp(-u))
    blob = [i for i in range(m) if i not in net.leaf_edges().values()]
    def unpack(u):
        y = sig(u[m:2*m]); x = np.sqrt(y)*sig(u[:m]); d = sig(u[2*m:]); return x, y, d
    rows = []
    for t in range(150):
        s = least_squares(lambda u: log_e(net.numeric(*unpack(u))), rng.normal(0, 2, 2*m+r), method='trf',
                          xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
        if np.linalg.norm(s.fun) > 1e-9: continue
        x, y, d = unpack(s.x)
        gaps = {}
        for i in blob:
            gaps[f'X{i}->0'] = x[i]; gaps[f'X{i}^2->Y{i}'] = y[i] - x[i]**2; gaps[f'Y{i}->1'] = 1 - y[i]; gaps[f'Y{i}->0'] = y[i]
        for j in range(r):
            gaps[f'd{j}->0'] = d[j]; gaps[f'd{j}->1'] = 1 - d[j]
        tight = sorted(gaps.items(), key=lambda kv: kv[1])[:3]
        rows.append((np.linalg.norm(s.fun), tight))
    print(f'L2-{idx}: near-zeros found {len(rows)}; tightest constraints at each:')
    for res, tight in sorted(rows)[:12]:
        print(f'   |e|={res:.1e}  ', ', '.join(f'{k}: {v:.1e}' for k, v in tight))
