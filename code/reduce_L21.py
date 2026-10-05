"""Check the reduced (separated) form of the L2-1 tree conditions numerically and
sample how often the Y/delta-level feasibility conditions hold."""
import json, numpy as np, sympy as sp
from k2p_model import *
c = json.load(open('../data/trinets_level2_unlabelled.json'))[1]
net = K2PNetwork(c['sd'], 'L2-1')
X0,X1,X2,X3,X4,X8 = [net.X[i] for i in (0,1,2,3,4,8)]
Y0,Y1,Y2,Y3,Y4,Y8 = [net.Y[i] for i in (0,1,2,3,4,8)]
d0,d1 = net.d
leaf = net.leaf_edges(); sub = {net.X[i]:1 for i in leaf.values()}; sub.update({net.Y[i]:1 for i in leaf.values()})
co = {k: sp.expand(v.subs(sub)) for k,v in net.coords.items()}
s1,s2,s3 = sp.symbols('s1 s2 s3')
U1 = d0*Y0 + (1-d0)*Y4*Y8; V1 = d0*Y0*Y3 + (1-d0)*Y8
U2 = d1*Y1 + (1-d1)*Y2*Y4; V2 = d1*Y1*Y3 + (1-d1)*Y2
# identities: C_i - s_i A_i in separated form
id1 = sp.expand(co['C1'] - s1*co['A1'] - (d1*X1*X3*(U1-s1) + (1-d1)*X2*X4*(V1-s1)))
id2 = sp.expand(co['C2'] - s2*co['A2'] - (d0*X0*X3*(U2-s2) + (1-d0)*X4*X8*(V2-s2)))
id3 = sp.expand(co['C3'] - s3*co['A3'] - (d0*d1*X0*X1*(Y3-s3) + d0*(1-d1)*X0*X2*X3*X4*(1-s3)
                                          + (1-d0)*d1*X1*X3*X4*X8*(1-s3) + (1-d0)*(1-d1)*X2*X8*(Y4-s3)))
print('separated-form identities hold:', id1==0, id2==0, id3==0)
idB = [sp.expand(co['B1'] - (d1*Y1*Y3 + (1-d1)*Y2*Y4)), sp.expand(co['B2'] - (d0*Y0*Y3 + (1-d0)*Y4*Y8))]
print('B1,B2 forms hold:', [e==0 for e in idB])
# sampling feasibility of Y/delta-level conditions
rng = np.random.default_rng(3); N=10**6
Y = rng.uniform(0,1,(6,N)); d = rng.uniform(0,1,(2,N))
y0,y1,y2,y3,y4,y8 = Y; D0,D1 = d
B1 = D1*y1*y3 + (1-D1)*y2*y4; B2 = D0*y0*y3 + (1-D0)*y4*y8
B3 = D0*D1*y0*y1 + D0*(1-D1)*y0*y2*y3*y4 + (1-D0)*D1*y1*y3*y4*y8 + (1-D0)*(1-D1)*y2*y8
S1=np.sqrt(B2*B3/B1); S2=np.sqrt(B1*B3/B2); S3=np.sqrt(B1*B2/B3)
u1 = D0*y0 + (1-D0)*y4*y8; v1 = D0*y0*y3 + (1-D0)*y8
u2 = D1*y1 + (1-D1)*y2*y4; v2 = D1*y1*y3 + (1-D1)*y2
c1 = (S1-u1)*(S1-v1) < 0; c2 = (S2-u2)*(S2-v2) < 0; c3 = (S3 > y3) | (S3 > y4)
print('P(cond1 betweenness)=%.4f P(cond2)=%.4f P(cond3 sign room)=%.4f P(all)=%.4f' % (c1.mean(), c2.mean(), c3.mean(), (c1&c2&c3).mean()))
