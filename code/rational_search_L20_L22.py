"""
All-rational K2P counterexample searches for the level-2 trinets L2-0 and L2-2 (region R3: X_e <= Y_e).
Same strategy as for L2-1: fix the star-tree G-parameters y1, y2, y3 rationally, solve the (linear)
B-equations B1 = y2 y3, B2 = y1 y3, B3 = y1 y2 for three Y's, reduce C_i = y_i A_i to ratio conditions
plus one conic, find a rational point on the conic.  Every candidate is re-verified exactly with the
generic symbolic model (k2p_model.K2PNetwork) before being reported.
"""
import random, math, json, sys
from fractions import Fraction as F
import sympy as sp
from k2p_model import K2PNetwork

which = sys.argv[1]            # 'L2-0' or 'L2-2'
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
ntrials = int(sys.argv[3]) if len(sys.argv) > 3 else 200000
REGION = sys.argv[4] if len(sys.argv) > 4 else 'R3'
cap = {'R1': lambda y: (1 + y)/2, 'R3': lambda y: y}[REGION]
import collections
stage = collections.Counter()

H = lambda v: max(abs(v.numerator), v.denominator)
def sqrtF(q):
    if q <= 0: return None
    a, b = math.isqrt(q.numerator), math.isqrt(q.denominator)
    return F(a, b) if a*a == q.numerator and b*b == q.denominator else None
g10 = [F(k, 10) for k in range(1, 10)]; g20 = [F(k, 20) for k in range(1, 20)]
smallQ = sorted({F(p, q) for q in range(1, 41) for p in range(1, 3*q)}, key=H)

idx = {'L2-0': 0, 'L2-2': 2}[which]
c = json.load(open('../data/trinets_level2_unlabelled.json'))[idx]
net = K2PNetwork(c['sd'], which)
leaf = net.leaf_edges()

def verify(par):
    sub = {}
    for e in range(len(net.sd)):
        if e in leaf.values():
            sub[net.X[e]] = 1; sub[net.Y[e]] = 1
        else:
            sub[net.X[e]] = sp.Rational(str(par[f'X{e}'])); sub[net.Y[e]] = sp.Rational(str(par[f'Y{e}']))
    sub[net.d[0]] = sp.Rational(str(par['d0'])); sub[net.d[1]] = sp.Rational(str(par['d1']))
    return [sp.nsimplify(e.subs(sub)) for e in net.E]

from sympy.solvers.diophantine.diophantine import diop_ternary_quadratic
_xs, _ys, _zs = sp.symbols('x y z', integer=True)
def conic_points(a, b, c, limit=60):
    """rational points (s, w), s, w > 0, on a s^2 + b + c w^2 = 0 (small-height scan, then Legendre + lines)"""
    out = []
    for s0 in smallQ[:600]:
        if c == 0: break
        w0 = sqrtF(-(b + a*s0*s0)/c)
        if w0 is not None:
            out.append((s0, w0))
            if len(out) >= limit: return out
    if out: base = out[0]
    else:
        L = math.lcm(a.denominator, b.denominator, c.denominator)
        A, B, C = int(a*L), int(b*L), int(c*L)
        if 0 in (A, B, C) or max(abs(A), abs(B), abs(C)) > 10**16: return out
        try: sol = diop_ternary_quadratic(A*_xs**2 + C*_ys**2 + B*_zs**2)   # x=S, y=W, z=T
        except Exception: return out
        if sol is None or sol[0] is None or sol[2] == 0: return out
        S, W, T = [int(v) for v in sol]
        if A*S*S + C*W*W + B*T*T != 0: return out          # guard against solver glitches
        base = (F(abs(S), abs(T)), F(abs(W), abs(T)))
        out.append(base)
    s0, w0 = base
    for q in range(1, 13):
        for p in range(-40, 41):
            m = F(p, q); dd = a + c*m*m
            if dd == 0: continue
            t = -2*(a*s0 + c*w0*m)/dd
            s1, w1 = abs(s0 + t), abs(w0 + m*t)
            if s1 > 0 and w1 > 0: out.append((s1, w1))
    out.sort(key=lambda sw: max(H(sw[0]), H(sw[1])))
    return out[:limit*4]

sols = []
for trial in range(ntrials):
    d0, d1 = random.choice(g10), random.choice(g10)
    y1, y2, y3 = random.choice(g20), random.choice(g20), random.choice(g20)
    if which == 'L2-0':
        Y1, Y3, Y4 = random.choice(g20), random.choice(g20), random.choice(g20)
        Y2 = (y2*y3/Y4 - d1*Y1*Y3)/(1 - d1)
        if not 0 < Y2 < 1: continue
        a11, a12, b1 = d0*Y3*Y4, (1 - d0), y1*y3
        a21 = d0*d1*Y1 + d0*(1 - d1)*Y2*Y3; a22 = (1 - d0)*d1*Y1*Y3*Y4 + (1 - d0)*(1 - d1)*Y2*Y4; b2 = y1*y2
        det = a11*a22 - a12*a21
        if det == 0: continue
        Y0 = (b1*a22 - a12*b2)/det; Y8 = (a11*b2 - a21*b1)/det
        if not (0 < Y0 < 1 and 0 < Y8 < 1): continue
        U1 = d0*Y0 + (1 - d0)*Y8; V1 = d0*Y0*Y3 + (1 - d0)*Y8
        U2 = d1*Y1 + (1 - d1)*Y2; V2 = Y4*(d1*Y1*Y3 + (1 - d1)*Y2)
        stage['Y-ok'] += 1
        if not ((y1 - U1)*(y1 - V1) < 0 and (y2 - U2)*(y2 - V2) < 0): continue
        stage['between'] += 1
        rho1 = (1 - d1)*(y1 - V1)/(d1*(U1 - y1))      # X1 X3 = rho1 X2
        rho2 = (1 - d0)*(y2 - V2)/(d0*(U2 - y2))      # X0 X3 X4 = rho2 X8
        al = d0*d1*rho1*rho2*(Y3*Y4 - y3); be = d0*(1 - d1)*rho2*(Y4 - y3); ga = (1 - d0)*(d1*rho1 + 1 - d1)*(1 - y3)
        for s, w in conic_points(al, be, ga):          # s = 1/X3, w = X4
            if s == 0 or w == 0: continue
            X3, X4 = 1/s, w
            if not (X3 < cap(Y3) and X4 < cap(Y4)): continue
            X2 = next((x for x in smallQ if x < cap(Y2) and rho1*x*s < cap(Y1)), None)
            X8 = next((x for x in smallQ if x < cap(Y8) and rho2*x*s/w < cap(Y0)), None)
            if X2 is None or X8 is None: continue
            par = dict(d0=d0, d1=d1, X0=rho2*X8*s/w, X1=rho1*X2*s, X2=X2, X3=X3, X4=X4, X8=X8,
                       Y0=Y0, Y1=Y1, Y2=Y2, Y3=Y3, Y4=Y4, Y8=Y8, y1=y1, y2=y2, y3=y3)
            sols.append(par); break
    else:  # L2-2
        Y0, Y3, Y4 = random.choice(g20), random.choice(g20), random.choice(g20)
        Y8 = (y2*y3/Y4 - d0*Y0*Y3)/(1 - d0)
        if not 0 < Y8 < 1: continue
        a11, a12, b1 = d1*Y3, d0*(1 - d1)*Y0*Y3*Y4 + (1 - d0)*(1 - d1)*Y4*Y8, y1*y3
        a21, a22, b2 = d0*d1*Y0*Y4 + (1 - d0)*d1*Y3*Y4*Y8, (1 - d1), y1*y2
        det = a11*a22 - a12*a21
        if det == 0: continue
        Y1 = (b1*a22 - a12*b2)/det; Y2 = (a11*b2 - a21*b1)/det
        if not (0 < Y1 < 1 and 0 < Y2 < 1): continue
        U1 = d1*Y1 + (1 - d1)*Y2; V1 = d1*Y1*Y3 + (1 - d1)*Y2
        stage['Y-ok'] += 1
        if not ((y1 - U1)*(y1 - V1) < 0): continue
        stage['between'] += 1
        rho1 = (1 - d0)*(y1 - V1)/(d0*(U1 - y1))      # X0 X3 = rho1 X8
        W2 = d0*Y0*Y4 + (1 - d0)*Y4*Y8; W3 = d0*Y0*Y3*Y4 + (1 - d0)*Y4*Y8
        K = (1 - y2)*(d0*rho1 + 1 - d0)
        N = (W2 - y2)*(W3 - y3)
        # Z^2 = X3^2 N / (K (d0 rho1 (Y3-y3) + (1-d0)(1-y3) X3^2)),  Z = X4 X8
        aa = N*K*d0*rho1*(Y3 - y3); bb = N*K*(1 - d0)*(1 - y3)
        done = False
        for X3, v in conic_points(bb, aa, F(-1)):
            if done: break
            if not X3 < cap(Y3): continue
            Dd = K*(d0*rho1*(Y3 - y3) + (1 - d0)*(1 - y3)*X3*X3)
            if Dd <= 0 or N/Dd <= 0: continue
            Z = sqrtF(X3*X3*N/Dd)
            if Z is None: continue
            # split Z = X4 X8 and choose X1; X2 = X1 kappa', X0 = rho1 X8 / X3
            for X4 in smallQ:
                X8 = Z/X4
                if not (X4 < cap(Y4) and X8 < cap(Y8)): continue
                X0 = rho1*X8/X3
                if not X0 < cap(Y0): continue
                kap = -d1*X4*X8*(d0*rho1*(Y3 - y3)/X3 + (1 - d0)*X3*(1 - y3))/((1 - d1)*(W3 - y3))
                if kap <= 0: break
                X1 = next((x for x in smallQ if x < cap(Y1) and x*kap < cap(Y2)), None)
                if X1 is None: break
                par = dict(d0=d0, d1=d1, X0=X0, X1=X1, X2=X1*kap, X3=X3, X4=X4, X8=X8,
                           Y0=Y0, Y1=Y1, Y2=Y2, Y3=Y3, Y4=Y4, Y8=Y8, y1=y1, y2=y2, y3=y3)
                sols.append(par); done = True; break
    if len(sols) >= 25: break

good = []
for par in sols:
    E = verify(par)
    ok = all(e == 0 for e in E)
    reg = all(0 < par[f'X{e}'] < cap(par[f'Y{e}']) and par[f'Y{e}'] < 1 for e in (0, 1, 2, 3, 4, 8)) and 0 < par['d0'] < 1 and 0 < par['d1'] < 1
    if ok and reg:
        good.append((max(H(v) for v in par.values()), par))
good.sort(key=lambda t: t[0])
for h, par in good[:5]:
    print(which, 'height', h, {k: str(v) for k, v in par.items()})
json.dump([dict(height=h, **{k: str(v) for k, v in par.items()}) for h, par in good],
          open(f'../data/rational_{which}_{REGION}_seed{sys.argv[2] if len(sys.argv) > 2 else 1}.json', 'w'), indent=1)
print('stages', dict(stage)); print(which, 'candidates', len(sols), 'verified exactly with E=0 and in region R3:', len(good))
