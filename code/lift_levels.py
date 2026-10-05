"""
Propagation to higher level (Proposition D(b)).  Starting from L2-1, repeatedly add a reticulation:
subdivide a blob edge e by a new tree vertex s and the pendant edge of leaf 3 by a new vertex v, and add
the arc s -> v (v becomes a reticulation with leaf 3 below it).  Deleting s -> v gives back the previous
network.  For each new network: check it is a reduced trinet with a strict level-k 3-blob, then continue
the certified L2-1 coincidence point by Newton/least squares at new mixing weight delta = 0.95, inside the
embeddable, transition-biased region (X <= Y on every edge).
"""
import json, numpy as np
from scipy.optimize import least_squares
from trinets import analyse_sd
from k2p_model import K2PNetwork, log_e
from fractions import Fraction as F
log = open('../logs/lift_levels.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()

cert = json.load(open('../certificates/L2-1_example.json'))
sd = [tuple(e) for e in cert['sd']]
par = {i: (float(F(cert['params'][str(i)][0])), float(F(cert['params'][str(i)][1]))) for i in range(len(sd))}
dl = {'h0': 0.5, 'h1': 0.5}

def add_reticulation(sd, par, dl, blob_edge_idx, k):
    u, w, d = sd[blob_edge_idx]
    s, v = f's{k}', f'v{k}'
    leaf3 = [i for i, e in enumerate(sd) if '3' in e[:2]][0]
    a3 = sd[leaf3][0] if sd[leaf3][1] == '3' else sd[leaf3][1]
    new = []
    newpar = {}
    for i, e in enumerate(sd):
        if i == blob_edge_idx or i == leaf3:
            continue
        newpar[len(new)] = par[i]; new.append(e)
    # split blob edge u - w (keeping its orientation flag on the lower half) : u - s - w
    Xe, Ye = par[blob_edge_idx]; fx, fy = 0.95, 0.97
    newpar[len(new)] = (Xe/fx, Ye/fy); new.append((u, s, False))
    newpar[len(new)] = (fx, fy); new.append((s, w, d))
    # split pendant edge a3 - 3 : a3 -> v (reticulation arc) , v - 3
    Xl, Yl = par[leaf3]
    newpar[len(new)] = (fx, fy); new.append((a3, v, True))
    newpar[len(new)] = (Xl/fx, Yl/fy); new.append((v, '3', False))
    newpar[len(new)] = (0.3, 0.6); new.append((s, v, True))
    dl = dict(dl); dl[v] = 0.95       # weight on the first listed incoming arc of v, i.e. a3 -> v
    return new, newpar, dl

for k in (3, 4):
    blob_edges = [i for i, e in enumerate(sd) if '1' not in e[:2] and '2' not in e[:2] and '3' not in e[:2]]
    chosen = None
    for bi in blob_edges:
        cand, cpar, cdl = add_reticulation(sd, par, dl, bi, k)
        info = analyse_sd([list(e) for e in cand])
        if info and not info.get('tree') and info['ret'] == k:
            chosen = (bi, cand, cpar, cdl); break
    assert chosen, 'no valid construction'
    bi, sd, par, dl = chosen
    P(f'level {k}: subdivided blob edge {bi}; network edges {sd}; reduced strict level-{k} 3-blob: True')
    net = K2PNetwork(sd, f'lift{k}')
    m = len(sd); r = len(net.rets)
    x0 = np.array([par[i][0] for i in range(m)]); y0 = np.array([par[i][1] for i in range(m)])
    d0 = np.array([dl[h] for h in net.rets])
    d1 = d0.copy(); d1[net.rets.index(f'v{k}')] = 1.0
    P(f'   |e| with new delta = 1 (should vanish): {np.linalg.norm(log_e(net.numeric(x0, y0, d1))):.3e}')
    P(f'   |e| at the continued starting point (new delta=0.95, old parameters): {np.linalg.norm(log_e(net.numeric(x0, y0, d0))):.3e}')
    sig = lambda u: 1/(1+np.exp(-u)); logit = lambda p: np.log(p/(1-p))
    jv = net.rets.index(f'v{k}')
    u = np.concatenate([logit(x0/y0), logit(y0), logit(np.delete(d0, jv))])
    def unpack(u, dv):
        y = sig(u[m:2*m]); x = y*sig(u[:m]); d = np.insert(sig(u[2*m:]), jv, dv); return x, y, d
    u_start = u.copy()
    for dv in np.linspace(1.0, 0.9, 21):          # continuation in the new mixing weight, fixed at each step
        sol = least_squares(lambda uu: log_e(net.numeric(*unpack(uu, dv))), u, method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15) \
            if False else least_squares(lambda uu: log_e(net.numeric(*unpack(uu, dv))), u, method='trf', x_scale=1.0,
                                        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=200)
        u = sol.x
    x, y, d = unpack(u, 0.9)
    res = np.linalg.norm(log_e(net.numeric(x, y, d)))
    P(f'   continuation to new delta = 0.9: |e| = {res:.3e}; max X/Y = {np.max(x/y):.3f}; Y in [{y.min():.3f},{y.max():.3f}]; '
      f'deltas {np.round(d,3).tolist()}; max change in logit-parameters {np.max(np.abs(u-u_start)):.3f}')
    par = {i: (x[i], y[i]) for i in range(m)}; dl = {h: d[j] for j, h in enumerate(net.rets)}
    json.dump(dict(sd=sd, X=x.tolist(), Y=y.tolist(), delta=dict(zip(net.rets, d.tolist()))),
              open(f'../data/lift_level{k}_zero.json', 'w'), indent=1)
