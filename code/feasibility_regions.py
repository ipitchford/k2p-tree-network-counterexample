"""
Feasibility of tree-network intersections for the non-stacked level-2 trinets, via the exact reductions
(see rational_search_*.py): sample (Y, delta) in (0,1)^8, compute the star-tree G-parameters
y_i = sqrt(B_j B_k / B_i), check the betweenness conditions, then decide (by a 1-D scan) whether the
remaining X-conditions can be met with every blob X_e below the cap c(Y_e) of the region:
  R1 = Theta0: c(Y) = (1+Y)/2;  R2 = embeddable: c(Y) = sqrt(Y);  R3: c(Y) = Y.
Free X's that only enter through ratios are scaled down (always possible), so only X3, X4 (and X8 for L2-2)
are constrained.  Reports the fraction of (Y, delta) samples admitting a completion.
"""
import numpy as np, sys
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400000
caps = {'R1': lambda y: (1 + y) / 2, 'R2': np.sqrt, 'R3': lambda y: y}
grid = np.linspace(1e-4, 1, 400)

def B_L20(Y0, Y1, Y2, Y3, Y4, Y8, d0, d1):
    B1 = Y4*(d1*Y1*Y3 + (1-d1)*Y2); B2 = d0*Y0*Y3*Y4 + (1-d0)*Y8
    B3 = d0*d1*Y0*Y1 + d0*(1-d1)*Y0*Y2*Y3 + (1-d0)*d1*Y1*Y3*Y4*Y8 + (1-d0)*(1-d1)*Y2*Y4*Y8
    return B1, B2, B3
def B_L21(Y0, Y1, Y2, Y3, Y4, Y8, d0, d1):
    B1 = d1*Y1*Y3 + (1-d1)*Y2*Y4; B2 = d0*Y0*Y3 + (1-d0)*Y4*Y8
    B3 = d0*d1*Y0*Y1 + d0*(1-d1)*Y0*Y2*Y3*Y4 + (1-d0)*d1*Y1*Y3*Y4*Y8 + (1-d0)*(1-d1)*Y2*Y8
    return B1, B2, B3
def B_L22(Y0, Y1, Y2, Y3, Y4, Y8, d0, d1):
    B1 = Y4*(d0*Y0*Y3 + (1-d0)*Y8)
    B2 = d0*(1-d1)*Y0*Y2*Y3*Y4 + d1*Y1*Y3 + (1-d0)*(1-d1)*Y2*Y4*Y8
    B3 = d0*d1*Y0*Y1*Y4 + (1-d0)*d1*Y1*Y3*Y4*Y8 + (1-d1)*Y2
    return B1, B2, B3

def feasible(shape, Y, d, c):
    Y0, Y1, Y2, Y3, Y4, Y8 = Y; d0, d1 = d
    B1, B2, B3 = {'L2-0': B_L20, 'L2-1': B_L21, 'L2-2': B_L22}[shape](Y0, Y1, Y2, Y3, Y4, Y8, d0, d1)
    y1, y2, y3 = np.sqrt(B2*B3/B1), np.sqrt(B1*B3/B2), np.sqrt(B1*B2/B3)
    ok = np.zeros_like(Y0, dtype=bool)
    if shape == 'L2-0':
        U1 = d0*Y0 + (1-d0)*Y8; V1 = d0*Y0*Y3 + (1-d0)*Y8; U2 = d1*Y1 + (1-d1)*Y2; V2 = Y4*(d1*Y1*Y3 + (1-d1)*Y2)
        bt = ((y1-U1)*(y1-V1) < 0) & ((y2-U2)*(y2-V2) < 0) & (y3 > Y3*Y4)
        rho1 = (1-d1)*(y1-V1)/(d1*(U1-y1)); rho2 = (1-d0)*(y2-V2)/(d0*(U2-y2)); K1 = d1*rho1 + 1 - d1
        G = c(Y3)**2*((1-d0)*K1*(1-y3)*c(Y4)**2 - d0*(1-d1)*rho2*(y3-Y4)) - d0*d1*rho1*rho2*(y3-Y3*Y4)
        ok = bt & (G > 0)
    elif shape == 'L2-1':
        U1 = d0*Y0 + (1-d0)*Y4*Y8; V1 = d0*Y0*Y3 + (1-d0)*Y8; U2 = d1*Y1 + (1-d1)*Y2*Y4; V2 = d1*Y1*Y3 + (1-d1)*Y2
        bt = ((y1-U1)*(y1-V1) < 0) & ((y2-U2)*(y2-V2) < 0)
        rho1 = (1-d1)*(y1-V1)/(d1*(U1-y1)); rho2 = (1-d0)*(y2-V2)/(d0*(U2-y2))
        kap = d0*(1-d1)*rho2 + (1-d0)*d1*rho1
        al = d0*d1*rho1*rho2*(Y3-y3); be = kap*(1-y3); ga = (1-d0)*(1-d1)*(Y4-y3)
        # need X4 <= c(Y4), X3 <= c(Y3) with al X4^2/X3^2 + be X4^2 + ga = 0; scan X3 = t*c(Y3)
        for t in grid:
            X3 = t*c(Y3)
            W = -ga/(al/X3**2 + be)
            ok |= bt & (W > 0) & (W <= c(Y4)**2)
    else:
        U1 = d1*Y1 + (1-d1)*Y2; V1 = d1*Y1*Y3 + (1-d1)*Y2
        bt = ((y1-U1)*(y1-V1) < 0)
        rho1 = (1-d0)*(y1-V1)/(d0*(U1-y1))
        W2 = d0*Y0*Y4 + (1-d0)*Y4*Y8; W3 = d0*Y0*Y3*Y4 + (1-d0)*Y4*Y8
        K = (1-y2)*(d0*rho1 + 1 - d0); Nn = (W2-y2)*(W3-y3)
        for t in grid:
            X3 = t*c(Y3)
            Dd = K*(d0*rho1*(Y3-y3) + (1-d0)*(1-y3)*X3**2)
            Z2 = X3**2*Nn/Dd
            kapnum = -(d0*rho1*(Y3-y3)/X3 + (1-d0)*X3*(1-y3))/((1-d1)*(W3-y3))
            Z = np.sqrt(np.abs(Z2))
            X8max = np.minimum(c(Y8), c(Y0)*X3/rho1)
            ok |= bt & (Z2 > 0) & (kapnum > 0) & (Z <= c(Y4)*X8max)
    return ok

for shape in ('L2-0', 'L2-1', 'L2-2'):
    Y = rng.uniform(0, 1, (6, N)); d = rng.uniform(0, 1, (2, N))
    res = {R: float(np.mean(feasible(shape, Y, d, c))) for R, c in caps.items()}
    print(shape, 'fraction of (Y,delta) samples admitting an intersection:', res, flush=True)
