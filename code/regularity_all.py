"""Exact Jacobian rank of (E1,E2,E3) and strict interiority at each level-2 certificate point."""
import json, sympy as sp
from fractions import Fraction as F
from k2p_model import K2PNetwork
for name in ('L2-0', 'L2-1', 'L2-2'):
    cert = json.load(open(f'../certificates/{name}_example.json'))
    net = K2PNetwork([tuple(e) for e in cert['sd']], name)
    sub = {}
    strict = True
    for k, (x, y) in cert['params'].items():
        X, Y = F(x), F(y); sub[net.X[int(k)]] = sp.Rational(x); sub[net.Y[int(k)]] = sp.Rational(y)
        strict &= (0 < X) and (0 < Y < 1) and (2*X < 1 + Y)
    rets = net.rets
    for j, h in enumerate(rets):
        sub[net.d[j]] = sp.Rational(cert['deltas'][h])
    J = sp.Matrix([[sp.diff(e, p).subs(sub) for p in net.params] for e in net.E])
    res = json.load(open(f'../certificates/{name}_example_result.json'))
    print(f'{name}: E at point = {[sp.simplify(e.subs(sub)) for e in net.E]}; Jacobian rank = {J.rank()}; '
          f'network strictly inside Theta0: {strict}; tree strictly admissible (Theta0): {res["tree_Theta0"]}')
