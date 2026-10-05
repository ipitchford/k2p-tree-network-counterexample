"""Numerical exploration: are the stacked level-2 trinets (L2-3, L2-4, L2-5) distinguishable from the
3-star tree under K3P on Theta0?  (K2P counterexamples for L2-0/1/2 are automatically K3P ones.)"""
import json, itertools, numpy as np
from scipy.optimize import least_squares
from trinets import gadd
from k2p_model import sd_displayed_trees
log = open('../logs/k3p_stacked.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
EL = {'C': 0, 'G': 1, 'T': 2}
labs = [s for s in itertools.product('ACGT', repeat=3) if gadd(*s) == 'A' and s != ('A','A','A')]
def model(sd):
    rets, inc, trees = sd_displayed_trees([tuple(e) for e in sd])
    m = len(sd)
    def q(a, d):          # a: (m,3) Fourier params for C,G,T ; d: (r,)
        out = {}
        for s in labs:
            tot = 0.0
            for ch, keep, splits in trees:
                w = 1.0
                for h, sg in ch.items():
                    w *= d[rets.index(h)] if sg == 0 else 1 - d[rets.index(h)]
                mon = 1.0
                for i in keep:
                    lab = gadd(*[s[int(l)-1] for l in splits[i]]) if splits[i] else 'A'
                    if lab != 'A': mon *= a[i, EL[lab]]
                tot += w * mon
            out[s] = tot
        return out
    return q, m, len(rets)
def resid(qd):
    # star-tree parameters from pair coordinates, then log-residuals of the 6 triple coordinates
    a = np.zeros((3, 3))
    for g in 'CGT':
        p12 = qd[(g, g, 'A')]; p13 = qd[(g, 'A', g)]; p23 = qd[('A', g, g)]
        a[0, EL[g]] = np.sqrt(p12*p13/p23); a[1, EL[g]] = np.sqrt(p12*p23/p13); a[2, EL[g]] = np.sqrt(p13*p23/p12)
    r = []
    for s in labs:
        if 'A' in s: continue
        r.append(np.log(qd[s]) - sum(np.log(a[i, EL[s[i]]]) for i in range(3)))
    return np.array(r)
sig = lambda u: 1/(1+np.exp(-u))
def unpack(u, m, r):
    # K3P Theta0 edge: barycentric-type parametrisation guaranteeing stochastic, positive definite
    # f(C), f(G), f(T) >= 0 with a_x in (0,1): sample transition probabilities f = (fA,fC,fG,fT) with fA largest
    z = u[:4*m].reshape(m, 4)
    w = np.exp(z); w = w / w.sum(axis=1, keepdims=True)
    fA, fC, fG, fT = w[:, 0], w[:, 1], w[:, 2], w[:, 3]
    aC = fA - fC + fG - fT; aG = fA + fC - fG - fT; aT = fA - fC - fG + fT
    a = np.stack([aC, aG, aT], axis=1)
    return a, sig(u[4*m:4*m+r])
rng = np.random.default_rng(5)
nets = json.load(open('../data/trinets_level2_unlabelled.json'))
for idx in range(6):
    q, m, r = model(nets[idx]['sd'])
    def F(u):
        a, d = unpack(u, m, r)
        if np.any(a <= 1e-9) or np.any(a >= 1): return 1e3*np.ones(6)
        return resid(q(a, d))
    best = []
    for t in range(40):
        u0 = rng.normal(0, 1, 4*m + r); u0[:4*m].reshape(m,4)[:, 0] += 2.0
        s = least_squares(F, u0, method='trf', xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=3000)
        a, d = unpack(s.x, m, r)
        best.append((np.linalg.norm(s.fun), float(a.min()), float(a.max()), float(min(d.min(), 1-d.max()))))
    best.sort()
    P(f'L2-{idx}: best K3P residual norms (residual, min a, max a, min dist of delta to 0/1):', [tuple(round(v, 6) if i else float(f'{v:.2e}') for i, v in enumerate(b)) for b in best[:4]])
