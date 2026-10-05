import json, sympy as sp, itertools
from k2p_model import *
for idx in (0, 2):
    c = json.load(open('../data/trinets_level2_unlabelled.json'))[idx]
    net = K2PNetwork(c['sd'], f'L2-{idx}')
    leaf = net.leaf_edges(); sub = {net.X[i]:1 for i in leaf.values()}; sub.update({net.Y[i]:1 for i in leaf.values()})
    co = {k: sp.expand(v.subs(sub)) for k,v in net.coords.items()}
    print(f'\n=== {net.name}: edges', list(enumerate(net.sd)), 'rets', net.rets)
    y1,y2,y3 = sp.symbols('y1 y2 y3')
    eqB = [co['B1']-y2*y3, co['B2']-y1*y3, co['B3']-y1*y2]
    eqC = [co['C1']-y1*co['A1'], co['C2']-y2*co['A2'], co['C3']-y3*co['A3']]
    for k in net.coord_names(): print(' ', k, '=', sp.factor(co[k]))
    blob = [p for p in net.params if p not in sub]
    eqs = eqB + eqC
    # search 6-subsets in which the whole system is jointly linear
    lin = []
    for S in itertools.combinations(blob, 6):
        ok = True
        for e in eqs:
            P = sp.Poly(e, *S)
            if P.total_degree() > 1: ok = False; break
        if ok:
            # also require the 6x6 coefficient matrix to be generically nonsingular
            M = sp.Matrix([[sp.Poly(e,*S).coeff_monomial(v) for v in S] for e in eqs])
            if sp.simplify(M.det()) != 0: lin.append(S)
    print('  jointly linear nonsingular 6-subsets:', lin[:10], 'count', len(lin))
