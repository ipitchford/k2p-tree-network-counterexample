import random, math
from fractions import Fraction as F
random.seed(3)
g10=[F(k,10) for k in range(1,10)]; g20=[F(k,20) for k in range(1,20)]
cnt={'start':0,'Y4<Y3':0,'Y1':0,'Y0Y8':0,'between':0,'X4sq>0':0,'X4<Y4':0}
import collections
for t in range(300000):
    cnt['start']+=1
    d0,d1=random.choice(g10),random.choice(g10)
    Y2,Y3,Y4=random.choice(g20),random.choice(g20),random.choice(g20)
    y1,y2=random.choice(g20),random.choice(g20)
    y3 = F(random.randint(1,19),20)   # free y3 this time
    Y1=(y2*y3-(1-d1)*Y2*Y4)/(d1*Y3)
    if not 0<Y1<1: continue
    cnt['Y1']+=1
    a11,a12,b1=d0*Y3,(1-d0)*Y4,y1*y3
    a21=d0*d1*Y1+d0*(1-d1)*Y2*Y3*Y4; a22=(1-d0)*d1*Y1*Y3*Y4+(1-d0)*(1-d1)*Y2; b2=y1*y2
    det=a11*a22-a12*a21
    if det==0: continue
    Y0=(b1*a22-a12*b2)/det; Y8=(a11*b2-a21*b1)/det
    if not(0<Y0<1 and 0<Y8<1): continue
    cnt['Y0Y8']+=1
    U1=d0*Y0+(1-d0)*Y4*Y8; V1=d0*Y0*Y3+(1-d0)*Y8; U2=d1*Y1+(1-d1)*Y2*Y4; V2=d1*Y1*Y3+(1-d1)*Y2
    if not((y1-U1)*(y1-V1)<0 and (y2-U2)*(y2-V2)<0): continue
    cnt['between']+=1
    if y3>Y3 or y3>Y4: cnt['Y4<Y3']+=1
print(cnt)
