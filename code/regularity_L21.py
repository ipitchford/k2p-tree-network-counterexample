"""Exact rank of the Jacobian of (E1,E2,E3) w.r.t. all network parameters at the L2-1 certificate point
(regularity needed for the propagation lemma)."""
import json, sympy as sp
from k2p_model import K2PNetwork
cert = json.load(open('../certificates/L2-1_example.json'))
net = K2PNetwork([tuple(e) for e in cert['sd']], 'L2-1')
sub = {}
for k, (x, y) in cert['params'].items():
    sub[net.X[int(k)]] = sp.Rational(x); sub[net.Y[int(k)]] = sp.Rational(y)
sub[net.d[0]] = sp.Rational(cert['deltas']['h0']); sub[net.d[1]] = sp.Rational(cert['deltas']['h1'])
E = [e for e in net.E]
print('E at point:', [sp.simplify(e.subs(sub)) for e in E])
J = sp.Matrix([[sp.diff(e, p).subs(sub) for p in net.params] for e in E])
print('Jacobian rank (exact):', J.rank())
# rank using blob parameters only (leaf edges excluded) and using only the 12 blob edge parameters (deltas fixed)
leaf = set(net.leaf_edges().values())
blob = [p for p in net.params if not any(p == net.X[i] or p == net.Y[i] for i in leaf)]
Jb = sp.Matrix([[sp.diff(e, p).subs(sub) for p in blob if p not in net.d] for e in E])
print('rank w.r.t. blob edge parameters only (deltas fixed):', Jb.rank())
