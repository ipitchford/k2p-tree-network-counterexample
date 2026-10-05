"""
Exact end-to-end certificate that a K2P trinet distribution equals a 3-star tree distribution.

Usage: python3 certify.py <certificate.json>
The JSON gives the semi-directed trinet (edge list), exact rational K2P parameters (X, Y) per edge and
reticulation weights.  The script
  1. computes the network's 64-entry leaf-pattern distribution exactly in probability space;
  2. derives the star-tree parameters from its Fourier coordinates (y_i = sqrt(B_j B_k / B_i),
     x_i = sqrt(A_j A_k / A_i));
  3. computes the star tree's distribution exactly (in Q(sqrt P) when needed);
  4. checks the two distributions are identical, entry by entry, and that every edge of both models
     satisfies the parameter conditions (Theta0; embeddable; transition rate >= transversion rate).
"""
import argparse, json, sys
from fractions import Fraction as F
from cert_common import *

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('certificate')
parser.add_argument('--output', help='Optional result path; inputs are never overwritten')
parser.add_argument('--require-region', choices=['Theta0', 'embeddable', 'kappa_ge_1'], default='Theta0')
args = parser.parse_args()
def require(condition, message):
    if not condition:
        raise ValueError(message)
cert = json.load(open(args.certificate))
sd = [tuple(e) for e in cert['sd']]
params = {int(k): (F(v[0]), F(v[1])) for k, v in cert['params'].items()}
deltas = {k: F(v) for k, v in cert['deltas'].items()}
out = {'name': cert['name']}

print(f"== {cert['name']}: {cert.get('description', '')}")
print('edges:', sd)
print('reticulation weights (first listed incoming edge):', {k: str(v) for k, v in deltas.items()})
# ---- network admissibility
net_adm = {i: admissible(*params[i], region='all') for i in params}
for i in sorted(params):
    print(f'  edge {i} {sd[i]}: X={params[i][0]}, Y={params[i][1]}  {net_adm[i]}')
for k in ('Theta0', 'embeddable', 'kappa_ge_1'):
    out[f'network_{k}'] = all(a[k] for a in net_adm.values())
out['deltas_in_(0,1)'] = all(0 < d < 1 for d in deltas.values())
require(out['deltas_in_(0,1)'], 'mixing weight outside (0,1)')
require(out['network_' + args.require_region], 'network outside required region')

# ---- 1. network distribution
pN = network_distribution(sd, params, deltas)
require(sum(pN.values()) == 1, 'network distribution does not sum to 1')
require(all(v > 0 for v in pN.values()), 'nonpositive probability')
qN = hadamard(pN)
require(qN['AAA'] == 1, 'Fourier normalization')
A = [qN['ACC'], qN['CAC'], qN['CCA']]; B = [qN['AGG'], qN['GAG'], qN['GGA']]; C = [qN['GCT'], qN['CGT'], qN['TCG']]
E = []
for i in range(3):
    j, k = [t for t in range(3) if t != i]
    E.append(A[i] ** 2 * B[j] * B[k] - B[i] * C[i] ** 2)
print('star-tree binomials E1, E2, E3 on the network point:', [str(e) for e in E])
out['E_all_zero'] = all(e == 0 for e in E)
require(out['E_all_zero'], 'tree binomial mismatch')

# ---- 2. star-tree parameters
PB = B[0] * B[1] * B[2]; rB = isqrtF(PB)
require(rB is not None, 'B1 B2 B3 is not a rational square')
y = [rB / B[i] for i in range(3)]
PA = A[0] * A[1] * A[2]; rA = isqrtF(PA)
if rA is not None:
    x = [rA / A[i] for i in range(3)]
    QS.P = F(1)
else:
    QS.P = PA
    x = [QS(0, 1 / A[i]) for i in range(3)]      # sqrt(PA)/A_i
print('star tree: y =', [str(v) for v in y])
print('           x =', [str(v) for v in x], f'  (P = A1 A2 A3 = {PA})')
print('           x (float) =', [float(v) if not isinstance(v, QS) else float(v) for v in x])
tree_adm = [admissible(x[i], y[i], region='all') for i in range(3)]
for i in range(3):
    print(f'  tree edge to leaf {i+1}: {tree_adm[i]}')
for k in ('Theta0', 'embeddable', 'kappa_ge_1'):
    out[f'tree_{k}'] = all(a[k] for a in tree_adm)
require(out['tree_' + args.require_region], 'comparison tree outside required region')

# ---- 3. star-tree distribution and comparison
star = [('c', '1'), ('c', '2'), ('c', '3')]
mats = [k2p_matrix(x[i], y[i]) for i in range(3)]
pT = tree_distribution(star, mats, ['1', '2', '3'], 'c')
mism = []
for pat in pN:
    t = pT[pat]
    t = t if isinstance(t, QS) else QS(t)
    if not (t.b == 0 and t.a == pN[pat]):
        mism.append(pat)
out['n_patterns'] = len(pN)
out['n_mismatches'] = len(mism)
out['distributions_identical'] = len(mism) == 0
require(len(pN) == 64 and not mism, 'complete probability distribution mismatch')
print(f'compared {len(pN)} leaf patterns; mismatches: {len(mism)}')
print('example probabilities  p(AAA) =', pN['AAA'], ' p(ACG) =', pN['ACG'], ' p(GTC) =', pN['GTC'])
print('RESULT:', json.dumps(out))
if args.output:
    json.dump(dict(out, network_distribution={k: str(v) for k, v in pN.items()},
                   tree_x=[str(v) for v in x], tree_y=[str(v) for v in y], P=str(PA)),
              open(args.output, 'w'), indent=2)
