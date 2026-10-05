"""
K2P Fourier parametrisation of 3-leaf semi-directed networks (displayed-tree model)
and the three star-tree binomials E1, E2, E3.

Notation (leaf positions i=1,2,3; {i,j,k} = {1,2,3}):
  A_i = q with state A at leaf i and C at the other two leaves   (q_ACC, q_CAC, q_CCA)
  B_i = q with state A at leaf i and G at the other two leaves   (q_AGG, q_GAG, q_GGA)
  C_i = q with state G at leaf i, C and T at the other two       (q_GCT, q_CGT, q_TCG)
  E_i = A_i^2 B_j B_k - q_AAA B_i C_i^2
E_3 is the polynomial Q of Lemma 3.? (lem:trinetk2p) of arXiv:2607.12919v3.
"""
import itertools
import networkx as nx
import numpy as np
import sympy as sp
from trinets import GROUP, NAME, LEAVES, gadd

STATES = 'ACGT'


def consistent_labellings():
    return [s for s in itertools.product(STATES, repeat=3) if gadd(*s) == 'A']


def sd_displayed_trees(sd):
    """sd: list of (u, v, directed_into_v). Returns list of (choice, tree_edges, splits)
    where splits[e] is the frozenset of leaves on the v-side of edge e=(u,v)."""
    rets = sorted({v for (u, v, d) in sd if d and sum(1 for e in sd if e[1] == v and e[2]) == 2})
    inc = {h: [i for i, e in enumerate(sd) if e[1] == h and e[2]] for h in rets}
    out = []
    for sigma in itertools.product((0, 1), repeat=len(rets)):
        drop = {inc[h][1 - s] for h, s in zip(rets, sigma)}
        keep = [i for i in range(len(sd)) if i not in drop]
        T = nx.Graph()
        for i in keep:
            T.add_edge(sd[i][0], sd[i][1], idx=i)
        assert nx.is_tree(T), 'displayed graph is not a spanning tree'
        splits = {}
        for i in keep:
            u, v, _ = sd[i]
            T2 = T.copy()
            T2.remove_edge(u, v)
            comp = nx.node_connected_component(T2, v)
            splits[i] = frozenset(l for l in LEAVES if l in comp)
        out.append((dict(zip(rets, sigma)), keep, splits))
    return rets, inc, out


class K2PNetwork:
    """Symbolic and numeric K2P model of a semi-directed trinet."""

    def __init__(self, sd, name=''):
        self.sd = [tuple(e) for e in sd]
        self.name = name
        self.rets, self.inc, self.trees = sd_displayed_trees(self.sd)
        m = len(self.sd)
        self.X = sp.symbols(f'X0:{m}', positive=True)
        self.Y = sp.symbols(f'Y0:{m}', positive=True)
        self.d = sp.symbols(f'd0:{len(self.rets)}', positive=True)
        self.q = {}
        for s in consistent_labellings():
            tot = 0
            for (ch, keep, splits) in self.trees:
                w = 1
                for h, sg in ch.items():
                    w *= self.d[self.rets.index(h)] if sg == 0 else (1 - self.d[self.rets.index(h)])
                mon = 1
                for i in keep:
                    lab = gadd(*[s[int(l) - 1] for l in splits[i]]) if splits[i] else 'A'
                    if lab in ('C', 'T'):
                        mon *= self.X[i]
                    elif lab == 'G':
                        mon *= self.Y[i]
                tot += w * mon
            self.q[''.join(s)] = sp.expand(tot)
        self.params = list(self.X) + list(self.Y) + list(self.d)
        self.coords = self.named_coords()
        self.E = self.binomials(self.coords)
        self._f = sp.lambdify(self.params, [self.coords[k] for k in self.coord_names()], 'numpy')

    @staticmethod
    def coord_names():
        return ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']

    def named_coords(self):
        q = self.q
        return {'A1': q['ACC'], 'A2': q['CAC'], 'A3': q['CCA'],
                'B1': q['AGG'], 'B2': q['GAG'], 'B3': q['GGA'],
                'C1': q['GCT'], 'C2': q['CGT'], 'C3': q['TCG'], 'AAA': q['AAA']}

    @staticmethod
    def binomials(c):
        A = [c['A1'], c['A2'], c['A3']]
        B = [c['B1'], c['B2'], c['B3']]
        C = [c['C1'], c['C2'], c['C3']]
        E = []
        for i in range(3):
            j, k = [x for x in range(3) if x != i]
            E.append(A[i] ** 2 * B[j] * B[k] - c['AAA'] * B[i] * C[i] ** 2)
        return E

    def leaf_edges(self):
        return {l: [i for i, e in enumerate(self.sd) if l in e[:2]][0] for l in LEAVES}

    def numeric(self, x, y, d):
        """x, y: arrays (m,) or (m, N); d: (r,) or (r, N). Returns dict of coordinate arrays."""
        vals = self._f(*list(x), *list(y), *list(d))
        return dict(zip(self.coord_names(), vals))


def log_e(c):
    """Normalised binomials e_i = log(A_i^2 B_j B_k) - log(B_i C_i^2) (q_AAA = 1)."""
    A = np.array([c['A1'], c['A2'], c['A3']], dtype=float)
    B = np.array([c['B1'], c['B2'], c['B3']], dtype=float)
    C = np.array([c['C1'], c['C2'], c['C3']], dtype=float)
    out = []
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        out.append(2 * np.log(A[i]) + np.log(B[j]) + np.log(B[k]) - np.log(B[i]) - 2 * np.log(C[i]))
    return np.array(out)


def sample_k2p_edges(rng, m, N, stochastic=True):
    """Uniform-ish samples from the K2P edge parameter set Theta_0:
    0 < Y < 1, 0 < X < 1, and (if stochastic) 2X <= 1 + Y."""
    y = rng.uniform(0, 1, size=(m, N))
    if stochastic:
        x = rng.uniform(0, 1, size=(m, N)) * (1 + y) / 2
    else:
        x = rng.uniform(0, 1, size=(m, N))
    return x, y
