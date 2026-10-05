"""
Small-height all-rational K2P counterexample search on L2-1 (free y3; Legendre conic for cond3).
See rational_search_L21_v2.py for the reformulation.  Region R3: 0 < X_e <= Y_e < 1 on every edge.
"""
import random, math, json, sys
from fractions import Fraction as F
from sympy import symbols
from sympy.solvers.diophantine.diophantine import diop_ternary_quadratic
random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
xs, ys, zs = symbols('x y z', integer=True)
def sqrtF(q):
    if q <= 0: return None
    a, b = math.isqrt(q.numerator), math.isqrt(q.denominator)
    return F(a, b) if a*a == q.numerator and b*b == q.denominator else None
H = lambda v: max(abs(v.numerator), v.denominator)
g10 = [F(k, 10) for k in range(1, 10)]; g20 = [F(k, 20) for k in range(1, 20)]
smallQ = sorted({F(p, q) for q in range(1, 31) for p in range(1, q)}, key=H)
sols = []; trials = 0
while trials < int(sys.argv[2]) if len(sys.argv) > 2 else 300000:
    trials += 1
    d0, d1 = random.choice(g10), random.choice(g10)
    Y2, Y3, Y4 = random.choice(g20), random.choice(g20), random.choice(g20)
    y1, y2, y3 = random.choice(g20), random.choice(g20), random.choice(g20)
    Y1 = (y2*y3 - (1-d1)*Y2*Y4)/(d1*Y3)
    if not 0 < Y1 < 1: continue
    a11, a12, b1 = d0*Y3, (1-d0)*Y4, y1*y3
    a21 = d0*d1*Y1 + d0*(1-d1)*Y2*Y3*Y4; a22 = (1-d0)*d1*Y1*Y3*Y4 + (1-d0)*(1-d1)*Y2; b2 = y1*y2
    det = a11*a22 - a12*a21
    if det == 0: continue
    Y0 = (b1*a22 - a12*b2)/det; Y8 = (a11*b2 - a21*b1)/det
    if not (0 < Y0 < 1 and 0 < Y8 < 1): continue
    U1 = d0*Y0 + (1-d0)*Y4*Y8; V1 = d0*Y0*Y3 + (1-d0)*Y8
    U2 = d1*Y1 + (1-d1)*Y2*Y4; V2 = d1*Y1*Y3 + (1-d1)*Y2
    if not ((y1-U1)*(y1-V1) < 0 and (y2-U2)*(y2-V2) < 0): continue
    rho1 = (1-d1)*(y1-V1)/(d1*(U1-y1)); rho2 = (1-d0)*(y2-V2)/(d0*(U2-y2))
    kappa = d0*(1-d1)*rho2 + (1-d0)*d1*rho1
    al = d0*d1*rho1*rho2*(Y3-y3); be = kappa*(1-y3); ga = (1-d0)*(1-d1)*(Y4-y3)
    if al >= 0 and ga >= 0: continue
    # quick small-height scan first: u small rational, w^2 = -(ga + al u^2)/be
    pt = None
    for u in smallQ[:400]:
        w = sqrtF(-(ga + al*u*u)/be)
        if w is not None:
            pt = (u, w); break
    if pt is None and al == 0: continue
    if pt is None:
        L = math.lcm(al.denominator, be.denominator, ga.denominator)
        a, b, c = int(al*L), int(be*L), int(ga*L)
        if max(abs(a), abs(b), abs(c)) > 10**14: continue
        try: s = diop_ternary_quadratic(a*xs**2 + b*ys**2 + c*zs**2)
        except Exception: continue
        if s is None or s[0] is None or s[2] == 0: continue
        pt = (F(int(s[0]), int(s[2])), F(int(s[1]), int(s[2])))
    u0, w0 = abs(pt[0]), abs(pt[1])
    if al*u0*u0 + be*w0*w0 + ga != 0:
        continue  # sympy Legendre output occasionally fails verification; skip
    # sweep lines through (u0, w0); keep the smallest-height admissible point
    bestpt = None
    for m in [F(p, q) for q in range(1, 16) for p in range(-80, 81)] + [None]:
        if m is None: u, w = u0, w0
        else:
            dd = al + be*m*m
            if dd == 0: continue
            t = -2*(al*u0 + be*w0*m)/dd; u, w = u0 + t, w0 + m*t
        if u <= 0 or w <= 0: continue
        X4, X3 = w, w/u
        if not (X4 < Y4 and X3 < Y3): continue
        h = max(H(X4), H(X3))
        if bestpt is None or h < bestpt[0]: bestpt = (h, X3, X4)
    if bestpt is None: continue
    _, X3, X4 = bestpt
    X2 = next((x for x in smallQ if x < Y2 and rho1*x*X4/X3 < Y1), None)
    X8 = next((x for x in smallQ if x < Y8 and rho2*x*X4/X3 < Y0), None)
    if X2 is None or X8 is None: continue
    X1 = rho1*X2*X4/X3; X0 = rho2*X8*X4/X3
    K1 = d1*rho1 + 1 - d1; K2 = d0*rho2 + 1 - d0
    Lq = d0*d1*rho1*rho2*(X4/X3)**2 + kappa*X4**2 + (1-d0)*(1-d1)
    rat = sqrtF(K1*K2*Lq) is not None
    par = dict(d0=d0, d1=d1, X0=X0, X1=X1, X2=X2, X3=X3, X4=X4, X8=X8, Y0=Y0, Y1=Y1, Y2=Y2, Y3=Y3, Y4=Y4, Y8=Y8, y1=y1, y2=y2, y3=y3)
    sols.append((not rat, max(H(v) for v in par.values()), par))
sols.sort(key=lambda s: (s[0], s[1]))
for nr, h, par in sols[:8]:
    print('rational tree x:', not nr, 'height', h, {k: str(v) for k, v in par.items()})
json.dump([dict(rational_tree_x=not nr, height=h, **{k: str(v) for k, v in par.items()}) for nr, h, par in sols[:50]],
          open(f'../data/rational_L21_v3_seed{sys.argv[1] if len(sys.argv)>1 else 5}.json', 'w'), indent=1)
print('trials', trials, 'solutions', len(sols), 'with rational tree x:', sum(1 for s in sols if not s[0]))
