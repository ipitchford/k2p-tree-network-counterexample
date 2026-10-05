"""
All-rational K2P counterexample search on the level-2 trinet L2-1 (small heights).

Key reformulation: E_1 = E_2 = E_3 = 0 (with positive coordinates) is equivalent to the existence of
positive y1, y2, y3 (the star-tree G-parameters) with
    B1 = y2 y3,  B2 = y1 y3,  B3 = y1 y2,              (bilinear in y)
    C_i = y_i A_i  (i = 1, 2, 3).                      (linear in y_i)
For L2-1 the B-equations are linear in (Y1) and then in (Y0, Y8) jointly, so choosing y1, y2, y3
rationally gives rational Y0, Y1, Y8.  Choosing y3 = Y3 kills the u-term of the remaining conic, so
cond3 reads beta X4^2 + gamma = 0 and only X4^2 has to be a rational square.
"""
import random, math, json, sys
from fractions import Fraction as F

random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 7)


def sqrtF(q):
    if q <= 0:
        return None
    n, d = q.numerator, q.denominator
    a, b = math.isqrt(n), math.isqrt(d)
    return F(a, b) if a * a == n and b * b == d else None


def height(v):
    return max(abs(v.numerator), v.denominator)


sols = []
trials = 0
grid10 = [F(k, 10) for k in range(1, 10)]
grid20 = [F(k, 20) for k in range(1, 20)]
while trials < 2_000_000 and len(sols) < 40:
    trials += 1
    d0, d1 = random.choice(grid10), random.choice(grid10)
    Y2, Y3, Y4 = random.choice(grid20), random.choice(grid20), random.choice(grid20)
    if not Y4 < Y3:
        continue
    y1, y2 = random.choice(grid20), random.choice(grid20)
    y3 = Y3
    Y1 = (y2 * y3 - (1 - d1) * Y2 * Y4) / (d1 * Y3)
    if not 0 < Y1 < 1:
        continue
    a11, a12, b1 = d0 * Y3, (1 - d0) * Y4, y1 * y3
    a21 = d0 * d1 * Y1 + d0 * (1 - d1) * Y2 * Y3 * Y4
    a22 = (1 - d0) * d1 * Y1 * Y3 * Y4 + (1 - d0) * (1 - d1) * Y2
    b2 = y1 * y2
    det = a11 * a22 - a12 * a21
    if det == 0:
        continue
    Y0 = (b1 * a22 - a12 * b2) / det
    Y8 = (a11 * b2 - a21 * b1) / det
    if not (0 < Y0 < 1 and 0 < Y8 < 1):
        continue
    U1 = d0 * Y0 + (1 - d0) * Y4 * Y8; V1 = d0 * Y0 * Y3 + (1 - d0) * Y8
    U2 = d1 * Y1 + (1 - d1) * Y2 * Y4; V2 = d1 * Y1 * Y3 + (1 - d1) * Y2
    if not ((y1 - U1) * (y1 - V1) < 0 and (y2 - U2) * (y2 - V2) < 0):
        continue
    rho1 = (1 - d1) * (y1 - V1) / (d1 * (U1 - y1))
    rho2 = (1 - d0) * (y2 - V2) / (d0 * (U2 - y2))
    kappa = d0 * (1 - d1) * rho2 + (1 - d0) * d1 * rho1
    X4sq = (1 - d0) * (1 - d1) * (y3 - Y4) / (kappa * (1 - y3))
    X4 = sqrtF(X4sq)
    if X4 is None or not X4 < Y4:
        continue
    # X3 free: try small rationals, prefer making A1A2A3 a square (rational star-tree x-parameters)
    K1 = d1 * rho1 + 1 - d1
    K2 = d0 * rho2 + 1 - d0
    best = None
    for X3 in sorted({F(p, q) for q in range(1, 41) for p in range(1, q)}, key=height):
        if not X3 < Y3:
            continue
        L = d0 * d1 * rho1 * rho2 * (X4 / X3) ** 2 + kappa * X4 ** 2 + (1 - d0) * (1 - d1)
        sq = sqrtF(K1 * K2 * L)
        if sq is not None:
            best = (X3, True)
            break
        if best is None:
            best = (X3, False)
    X3, rational_tree = best
    # free X2, X8 (small rationals) with X <= Y on edges 0, 1, 2, 8
    cand = None
    for X2 in sorted({F(p, q) for q in range(1, 41) for p in range(1, q)}, key=height):
        X1 = rho1 * X2 * X4 / X3
        if X2 < Y2 and X1 < Y1:
            cand = (X2, X1)
            break
    for X8 in sorted({F(p, q) for q in range(1, 41) for p in range(1, q)}, key=height):
        X0 = rho2 * X8 * X4 / X3
        if X8 < Y8 and X0 < Y0:
            break
    else:
        continue
    if cand is None:
        continue
    X2, X1 = cand
    par = dict(d0=d0, d1=d1, X0=X0, X1=X1, X2=X2, X3=X3, X4=X4, X8=X8,
               Y0=Y0, Y1=Y1, Y2=Y2, Y3=Y3, Y4=Y4, Y8=Y8, y1=y1, y2=y2, y3=y3)
    h = max(height(v) for v in par.values())
    sols.append((not rational_tree, h, par))

sols.sort(key=lambda s: (s[0], s[1]))
out = []
for nr, h, par in sols[:10]:
    print('rational tree x:', not nr, 'height', h, {k: str(v) for k, v in par.items()})
    out.append(dict(rational_tree_x=not nr, height=h, **{k: str(v) for k, v in par.items()}))
json.dump(out, open('../data/rational_L21_v2.json', 'w'), indent=1)
print('trials', trials, 'solutions', len(sols))
