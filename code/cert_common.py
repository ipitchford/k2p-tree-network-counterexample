"""
Independent exact certificate machinery (does NOT use k2p_model.py or any Fourier formula for the
network): leaf-pattern distributions are computed in probability space from 4x4 K2P transition matrices,
summing over all internal states of every displayed tree, with exact rational arithmetic (and exact
arithmetic in Q(sqrt P) for star-tree parameters that are square roots).

Nucleotides A, C, G, T are identified with Z2 x Z2 elements (0,0), (0,1), (1,0), (1,1) as in
arXiv:2607.12919. K2P edge parameters (X, Y) are the Fourier eigenvalues a_C = a_T = X, a_G = Y;
the transition matrix is M[a][b] = f(a xor b) with
    f(A) = (1 + 2X + Y)/4,  f(C) = (1 + Y - 2X)/4,  f(G) = f(T) = (1 - Y)/4.
"""
from fractions import Fraction as F
import itertools, math
import networkx as nx

NUC = 'ACGT'
GEL = {'A': (0, 0), 'C': (0, 1), 'G': (1, 0), 'T': (1, 1)}


class QS:
    """Exact numbers a + b*sqrt(P) with a, b rational and P a fixed positive rational."""
    P = None

    def __init__(self, a, b=F(0)):
        self.a, self.b = F(a), F(b)

    def __add__(self, o):
        o = o if isinstance(o, QS) else QS(o)
        return QS(self.a + o.a, self.b + o.b)
    __radd__ = __add__

    def __sub__(self, o):
        o = o if isinstance(o, QS) else QS(o)
        return QS(self.a - o.a, self.b - o.b)

    def __rsub__(self, o):
        return QS(o) - self

    def __mul__(self, o):
        o = o if isinstance(o, QS) else QS(o)
        return QS(self.a * o.a + self.b * o.b * QS.P, self.a * o.b + self.b * o.a)
    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, QS):
            raise TypeError('QS division requires a rational denominator')
        return QS(self.a / o, self.b / o)

    def __eq__(self, o):
        o = o if isinstance(o, QS) else QS(o)
        return self.a == o.a and self.b == o.b

    def __float__(self):
        return float(self.a) + float(self.b) * math.sqrt(float(QS.P))

    def __repr__(self):
        return f'({self.a} + {self.b}*sqrt({QS.P}))' if self.b else f'{self.a}'


def k2p_matrix(X, Y):
    fA = (1 + 2 * X + Y) / 4
    fC = (1 + Y - 2 * X) / 4
    fG = (1 - Y) / 4
    f = {'A': fA, 'C': fC, 'G': fG, 'T': fG}
    M = {}
    for a in NUC:
        for b in NUC:
            g = (GEL[a][0] ^ GEL[b][0], GEL[a][1] ^ GEL[b][1])
            M[(a, b)] = f[[k for k, v in GEL.items() if v == g][0]]
    return M


def tree_distribution(edges, mats, leaves, root):
    """edges: list of (u, v) undirected; mats[i]: K2P matrix for edges[i] (symmetric, so orientation
    is irrelevant); uniform root distribution. Returns dict leaf-pattern -> probability."""
    G = nx.Graph()
    for i, (u, v) in enumerate(edges):
        G.add_edge(u, v, i=i)
    if not nx.is_tree(G):
        raise ValueError('displayed graph is not a tree')
    order = list(nx.bfs_edges(G, root))
    internal = [v for v in G.nodes if v not in leaves]
    p = {}
    for pat in itertools.product(NUC, repeat=len(leaves)):
        fixed = dict(zip(leaves, pat))
        tot = 0
        for st in itertools.product(NUC, repeat=len(internal)):
            s = dict(zip(internal, st)); s.update(fixed)
            term = F(1, 4)
            for (u, v) in order:
                term = term * mats[G[u][v]['i']][(s[u], s[v])]
            tot = tot + term
        p[''.join(pat)] = tot
    return p


def displayed_trees(sd):
    """sd: list of (u, v, directed_into_v). Returns list of (weight-key tuple, kept edge indices)."""
    rets = sorted({v for (u, v, d) in sd if d and sum(1 for e in sd if e[1] == v and e[2]) == 2})
    inc = {h: [i for i, e in enumerate(sd) if e[1] == h and e[2]] for h in rets}
    out = []
    for sigma in itertools.product((0, 1), repeat=len(rets)):
        drop = {inc[h][1 - s] for h, s in zip(rets, sigma)}
        out.append((dict(zip(rets, sigma)), [i for i in range(len(sd)) if i not in drop]))
    return rets, inc, out


def network_distribution(sd, params, deltas, leaves=('1', '2', '3')):
    """params[i] = (X, Y) for edge i of sd; deltas[h] = weight of the FIRST listed incoming edge of h."""
    rets, inc, trees = displayed_trees(sd)
    mats = [k2p_matrix(*params[i]) for i in range(len(sd))]
    total = None
    for ch, keep in trees:
        w = F(1)
        for h, s in ch.items():
            w *= deltas[h] if s == 0 else (1 - deltas[h])
        edges = [(sd[i][0], sd[i][1]) for i in keep]
        root = [v for v in {x for e in edges for x in e} if v not in leaves][0]
        p = tree_distribution(edges, [mats[i] for i in keep], list(leaves), root)
        total = {k: w * v for k, v in p.items()} if total is None else {k: total[k] + w * p[k] for k in p}
    return total


def hadamard(p):
    """Fourier coordinates q_g = sum_x p_x prod_i (-1)^<g_i, x_i> (equals the product of edge
    eigenvalues for a tree with uniform root)."""
    q = {}
    for g in itertools.product(NUC, repeat=3):
        tot = 0
        for x, v in p.items():
            sgn = 1
            for gi, xi in zip(g, x):
                sgn *= (-1) ** (GEL[gi][0] * GEL[xi][0] + GEL[gi][1] * GEL[xi][1])
            tot = tot + sgn * v
        q[''.join(g)] = tot
    return q


def isqrtF(q):
    q = F(q)
    a, b = math.isqrt(q.numerator), math.isqrt(q.denominator)
    return F(a, b) if a * a == q.numerator and b * b == q.denominator else None


def admissible(X, Y, region='Theta0'):
    """Exact admissibility of a K2P edge (X, Y); X may be a QS of the form b*sqrt(P) (b > 0)."""
    def sq(x):          # x^2 as an exact rational
        if isinstance(x, QS):
            if not (x.a == 0 and x.b > 0):
                raise ValueError('unsupported or nonpositive quadratic number')
            return x.b * x.b * QS.P
        return F(x) * F(x)
    def pos(x):
        return (x.b > 0 and x.a == 0) if isinstance(x, QS) else x > 0
    Y = F(Y)
    ok = pos(X) and 0 < Y < 1
    ok_stoch = ok and sq(X) <= ((1 + Y) / 2) ** 2          # f(C) >= 0  <=>  X <= (1+Y)/2  (X > 0)
    ok_emb = ok and sq(X) <= Y                              # continuous-time K2P (alpha >= 0)
    ok_kappa = ok and sq(X) <= Y * Y                        # transition rate >= transversion rate
    return dict(Theta0=ok_stoch, embeddable=ok_stoch and ok_emb, kappa_ge_1=ok_stoch and ok_emb and ok_kappa)[region] \
        if region != 'all' else dict(Theta0=ok_stoch, embeddable=ok_stoch and ok_emb, kappa_ge_1=ok_stoch and ok_emb and ok_kappa)
