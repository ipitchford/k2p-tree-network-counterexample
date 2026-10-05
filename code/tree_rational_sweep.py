"""For each rational L2-1 solution family (fixed d, Y, y), sweep the rational points of the cond3 conic
and look for points where A1A2A3 is a rational square (=> rational star-tree x-parameters)."""
import json, math, glob, sys
from fractions import Fraction as F
def sqrtF(q):
    if q <= 0: return None
    a, b = math.isqrt(q.numerator), math.isqrt(q.denominator)
    return F(a, b) if a*a == q.numerator and b*b == q.denominator else None
H = lambda v: max(abs(v.numerator), v.denominator)
sols = json.load(open('../data/rational_L21_v3_seed5.json'))
hits = []
for S in sols:
    p = {k: F(v) for k, v in S.items() if k not in ('rational_tree_x', 'height')}
    d0, d1 = p['d0'], p['d1']; Y0, Y1, Y2, Y3, Y4, Y8 = [p[k] for k in ('Y0','Y1','Y2','Y3','Y4','Y8')]
    y1, y2, y3 = p['y1'], p['y2'], p['y3']
    U1 = d0*Y0 + (1-d0)*Y4*Y8; V1 = d0*Y0*Y3 + (1-d0)*Y8; U2 = d1*Y1 + (1-d1)*Y2*Y4; V2 = d1*Y1*Y3 + (1-d1)*Y2
    rho1 = (1-d1)*(y1-V1)/(d1*(U1-y1)); rho2 = (1-d0)*(y2-V2)/(d0*(U2-y2))
    kappa = d0*(1-d1)*rho2 + (1-d0)*d1*rho1
    al = d0*d1*rho1*rho2*(Y3-y3); be = kappa*(1-y3); ga = (1-d0)*(1-d1)*(Y4-y3)
    K1 = d1*rho1 + 1 - d1; K2 = d0*rho2 + 1 - d0
    u0, w0 = p['X4']/p['X3'], p['X4']
    assert al*u0*u0 + be*w0*w0 + ga == 0
    for q in range(1, 400):
        for pn in range(-1200, 1201):
            if math.gcd(pn, q) != 1: continue
            m = F(pn, q); dd = al + be*m*m
            if dd == 0: continue
            t = -2*(al*u0 + be*w0*m)/dd; u, w = u0 + t, w0 + m*t
            if u <= 0 or w <= 0 or not (w < Y4 and w/u < Y3): continue
            Lq = d0*d1*rho1*rho2*u*u + kappa*w*w + (1-d0)*(1-d1)
            if sqrtF(K1*K2*Lq) is not None:
                hits.append(dict(base=S, X3=str(w/u), X4=str(w), m=str(m)))
                print('HIT', S['height'], 'X3', w/u, 'X4', w, flush=True)
                break
        else: continue
        break
json.dump(hits, open('../data/tree_rational_hits.json', 'w'), indent=1)
print('hits', len(hits))
