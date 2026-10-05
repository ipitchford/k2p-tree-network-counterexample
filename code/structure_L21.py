import json, sympy as sp
from k2p_model import *
c = json.load(open('../data/trinets_level2_unlabelled.json'))[1]
net = K2PNetwork(c['sd'], 'L2-1')
print('edges:', list(enumerate(net.sd)))
leaf = net.leaf_edges(); print('leaf edges', leaf)
sub = {net.X[i]:1 for i in leaf.values()}; sub.update({net.Y[i]:1 for i in leaf.values()})
co = {k: sp.factor(sp.expand(v.subs(sub))) for k,v in net.coords.items()}
for k in net.coord_names(): print(k, '=', co[k])
E = [sp.expand(e.subs(sub)) for e in net.E]
blob = [p for p in net.params if p not in sub]
print('blob params', blob)
for p in blob:
    print(p, 'in coords:', [k for k in net.coord_names() if co[k].has(p)], ' deg in E1,E2,E3:', [sp.degree(e, p) for e in E])
