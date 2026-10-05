"""
Symbolic verification for the distinguishable ("stacked") level-2 trinets L2-3, L2-4, L2-5 and for the
3-sunlet.  For a leaf x below a reticulation r with incoming edges e' and e'',
    q_N = delta * q_{N - e''} + (1 - delta) * q_{N - e'},
so the displayed networks' coordinates are q_N at delta = 1 and delta = 0.  Coordinates with state A
at leaf x do not depend on delta (Lemma 2.? of arXiv:2607.12919), and
    E_x(N) = delta^2 E_x(N') + (1-delta)^2 E_x(N'') + delta(1-delta) * Cross,
with Cross >= 0 whenever E_x(N'), E_x(N'') >= 0 (AM-GM).  Here we check symbolically:
  (a) the coordinates with A at x are delta-free;
  (b) the decomposition identity;
  (c) for the stacked shapes, one displayed network gives E_x = 0 identically (a tree) and the other
      gives E_x = (positive) * (positive) on the parameter box (a 3-sunlet with reticulation leaf x).
"""
import json, sympy as sp
from k2p_model import K2PNetwork

log = open('../logs/good_shapes.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()

def coords_index(x):
    # E_x = A_x^2 B_j B_k - B_x C_x^2  with leaf positions 1..3
    i = int(x) - 1
    j, k = [t for t in range(3) if t != i]
    return i, j, k

def analyse(sd, name):
    net = K2PNetwork(sd, name)
    rets = net.rets
    # leaf adjacent to a reticulation whose other... choose the reticulation adjacent to a leaf
    cand = [(h, e[1] if e[0] == h else e[0]) for h in rets for e in net.sd
            if h in e[:2] and (e[1] in '123' or e[0] in '123')]
    P(f'\n=== {name}: edges {net.sd}; reticulations {rets}; leaf-adjacent reticulations {cand}')
    leaf = net.leaf_edges()
    sub1 = {net.X[i]: 1 for i in leaf.values()}; sub1.update({net.Y[i]: 1 for i in leaf.values()})
    for h, x in cand:
        d = net.d[rets.index(h)]
        i, j, k = coords_index(x)
        A = [net.coords['A1'], net.coords['A2'], net.coords['A3']]
        B = [net.coords['B1'], net.coords['B2'], net.coords['B3']]
        C = [net.coords['C1'], net.coords['C2'], net.coords['C3']]
        # (a)
        free = sp.expand(sp.diff(A[i], d)) == 0 and sp.expand(sp.diff(B[i], d)) == 0
        Ex = sp.expand(A[i]**2 * B[j] * B[k] - B[i] * C[i]**2)
        E1 = sp.expand(Ex.subs(d, 1)); E0 = sp.expand(Ex.subs(d, 0))
        Bj1, Bk1, Ci1 = [sp.expand(v.subs(d, 1)) for v in (B[j], B[k], C[i])]
        Bj0, Bk0, Ci0 = [sp.expand(v.subs(d, 0)) for v in (B[j], B[k], C[i])]
        cross = A[i]**2 * (Bj1 * Bk0 + Bj0 * Bk1) - 2 * B[i] * Ci1 * Ci0
        ident = sp.expand(Ex - (d**2 * E1 + (1 - d)**2 * E0 + d * (1 - d) * cross)) == 0
        P(f'  leaf x={x} below {h}: A_x,B_x free of delta: {free}; decomposition identity: {ident}')
        for lab, Ed in (('delta=1', E1), ('delta=0', E0)):
            f = sp.factor(Ed.subs(sub1))
            P(f'    E_x at {lab} (leaf edges stripped): {f}')

nets = json.load(open('../data/trinets_level2_unlabelled.json'))
sunlet = json.load(open('../data/trinets_level1_unlabelled.json'))[0]
analyse(sunlet['sd'], '3-sunlet')
for idx in (3, 4, 5):
    analyse(nets[idx]['sd'], f'L2-{idx}')
for idx in (0, 1, 2):
    analyse(nets[idx]['sd'], f'L2-{idx}')
