"""
Constructive search for an all-rational K2P counterexample on the level-2 trinet L2-1.

L2-1 blob (leaf edges removed):
  reticulation h0 (leaf 1) with parents t0 (edge 0, weight d0) and t1 (edge 8, weight 1-d0)
  reticulation h1 (leaf 2) with parents t0 (edge 1, weight d1) and t1 (edge 2, weight 1-d1)
  t0 -- t2 (edge 3), t1 -- t2 (edge 4), leaf 3 on t2.

Separated form of the star-tree conditions C_i = s_i A_i with s_i = sqrt(B_j B_k / B_i):
  cond1: d1 X1 X3 (U1 - s1) + (1-d1) X2 X4 (V1 - s1) = 0
  cond2: d0 X0 X3 (U2 - s2) + (1-d0) X4 X8 (V2 - s2) = 0
  cond3: d0 d1 X0 X1 (Y3-s3) + [d0(1-d1) X0 X2 + (1-d0) d1 X1 X8] X3 X4 (1-s3) + (1-d0)(1-d1) X2 X8 (Y4-s3) = 0
Construction:
  1. choose d0, d1, Y1, Y2, Y3, Y4, Y8 and a rational lambda; Y0 is then fixed so that B1 B2 B3 is a
     rational square (B2, B3 are affine in Y0, B1 does not involve Y0), so s1, s2, s3 are rational;
  2. rho1 = X1X3/(X2X4), rho2 = X0X3/(X4X8) are forced by cond1, cond2;
  3. cond3 becomes alpha u^2 + beta w^2 + gamma = 0 with u = X4/X3, w = X4 (a conic);
     find a rational point by Legendre's method and sweep the rational parametrisation;
  4. X2, X8 are free (cond3 does not depend on them); X1 = rho1 X2 X4/X3, X0 = rho2 X8 X4/X3.
Region R3 (embeddable, transition rate >= transversion rate): 0 < X_e <= Y_e < 1 on every edge.
"""
import random, json, math, sys
from fractions import Fraction as F
from sympy import symbols, Integer
from sympy.solvers.diophantine.diophantine import diop_ternary_quadratic

random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 1)


def isqrt_frac(q):
    """exact rational square root or None"""
    if q < 0:
        return None
    n, d = q.numerator, q.denominator
    rn, rd = math.isqrt(n), math.isqrt(d)
    if rn * rn == n and rd * rd == d:
        return F(rn, rd)
    return None


def rnd(lo=1, hi=19, den=20):
    return F(random.randint(lo, hi), den)


def blob_B(Y0, Y1, Y2, Y3, Y4, Y8, d0, d1):
    B1 = d1 * Y1 * Y3 + (1 - d1) * Y2 * Y4
    B2 = d0 * Y0 * Y3 + (1 - d0) * Y4 * Y8
    B3 = d0 * d1 * Y0 * Y1 + d0 * (1 - d1) * Y0 * Y2 * Y3 * Y4 + (1 - d0) * d1 * Y1 * Y3 * Y4 * Y8 + (1 - d0) * (1 - d1) * Y2 * Y8
    return B1, B2, B3


uu, ww, tt = symbols('x y z', integer=True)  # sympy returns solutions in sorted-variable order (x, y, z)
found = []
tries = 0
while len(found) < 5 and tries < 20000:
    tries += 1
    d0, d1 = rnd(1, 9, 10), rnd(1, 9, 10)
    Y1, Y2, Y3, Y4, Y8 = [rnd() for _ in range(5)]
    lam = F(random.randint(1, 30), random.randint(1, 30))
    # B2 = p Y0 + q, B3 = r Y0 + t ; B1 B2 B3 = lam^2 B2^2  <=>  B1 (r Y0 + t) = lam^2 (p Y0 + q)
    B1 = d1 * Y1 * Y3 + (1 - d1) * Y2 * Y4
    p, q = d0 * Y3, (1 - d0) * Y4 * Y8
    r = d0 * d1 * Y1 + d0 * (1 - d1) * Y2 * Y3 * Y4
    t = (1 - d0) * d1 * Y1 * Y3 * Y4 * Y8 + (1 - d0) * (1 - d1) * Y2 * Y8
    den = B1 * r - lam ** 2 * p
    if den == 0:
        continue
    Y0 = (lam ** 2 * q - B1 * t) / den
    if not (0 < Y0 < 1):
        continue
    B1, B2, B3 = blob_B(Y0, Y1, Y2, Y3, Y4, Y8, d0, d1)
    P = B1 * B2 * B3
    rt = isqrt_frac(P)
    assert rt is not None
    s1, s2, s3 = rt / B1, rt / B2, rt / B3          # s_i = sqrt(B1B2B3)/B_i = sqrt(B_jB_k/B_i)
    U1 = d0 * Y0 + (1 - d0) * Y4 * Y8; V1 = d0 * Y0 * Y3 + (1 - d0) * Y8
    U2 = d1 * Y1 + (1 - d1) * Y2 * Y4; V2 = d1 * Y1 * Y3 + (1 - d1) * Y2
    if not ((s1 - U1) * (s1 - V1) < 0 and (s2 - U2) * (s2 - V2) < 0):
        continue
    rho1 = (1 - d1) * (s1 - V1) / (d1 * (U1 - s1))
    rho2 = (1 - d0) * (s2 - V2) / (d0 * (U2 - s2))
    kappa = d0 * (1 - d1) * rho2 + (1 - d0) * d1 * rho1
    al = d0 * d1 * rho1 * rho2 * (Y3 - s3)
    be = kappa * (1 - s3)
    ga = (1 - d0) * (1 - d1) * (Y4 - s3)
    if be <= 0 or (al >= 0 and ga >= 0):
        continue
    # integer Legendre form a U^2 + b W^2 + c T^2 = 0
    L = math.lcm(al.denominator, be.denominator, ga.denominator)
    a, b, c = int(al * L), int(be * L), int(ga * L)
    g = math.gcd(math.gcd(a, b), c); a, b, c = a // g, b // g, c // g
    if max(abs(a), abs(b), abs(c)) > 10 ** 30:
        continue
    try:
        sol = diop_ternary_quadratic(a * uu ** 2 + b * ww ** 2 + c * tt ** 2)
    except Exception:
        continue
    if sol is None or sol[0] is None:
        continue
    U0, W0, T0 = [int(v) for v in sol]
    if T0 == 0:
        continue
    u0, w0 = F(U0, T0), F(W0, T0)
    assert al * u0 ** 2 + be * w0 ** 2 + ga == 0
    # sweep the rational parametrisation through (u0, w0): u = u0 + t, w = w0 + m t
    for mn in range(-60, 61):
        for md in range(1, 13):
            m = F(mn, md)
            dd = al + be * m * m
            if dd == 0:
                continue
            tpar = -2 * (al * u0 + be * w0 * m) / dd
            u, w = u0 + tpar, w0 + m * tpar
            if u <= 0 or w <= 0:
                continue
            X4, X3 = w, w / u
            if not (X4 < Y4 and X3 < Y3):
                continue
            # free X2, X8: make all X's satisfy X <= Y
            X2 = min(Y2, Y1 * X3 / (rho1 * X4)) * F(1, 2)
            X8 = min(Y8, Y0 * X3 / (rho2 * X4)) * F(1, 2)
            X1 = rho1 * X2 * X4 / X3
            X0 = rho2 * X8 * X4 / X3
            Xs = dict(X0=X0, X1=X1, X2=X2, X3=X3, X4=X4, X8=X8)
            Ys = dict(Y0=Y0, Y1=Y1, Y2=Y2, Y3=Y3, Y4=Y4, Y8=Y8)
            if all(0 < Xs['X%s' % k] < Ys['Y%s' % k] < 1 for k in ('0', '1', '2', '3', '4', '8')):
                height = max(max(v.numerator, v.denominator) for v in list(Xs.values()) + list(Ys.values()) + [d0, d1])
                found.append(dict(height=height, d0=d0, d1=d1, **Xs, **Ys, s=(s1, s2, s3)))
                break
        else:
            continue
        break

found.sort(key=lambda f: f['height'])
for f in found:
    print('height', f['height'], {k: str(v) for k, v in f.items() if k not in ('height', 's')}, 's=', [str(x) for x in f['s']])
json.dump([{k: (str(v) if not isinstance(v, tuple) else [str(x) for x in v]) for k, v in f.items()} for f in found],
          open(f'../data/rational_L21_seed{sys.argv[1] if len(sys.argv) > 1 else 1}.json', 'w'), indent=1)
print('tries', tries)
