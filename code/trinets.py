"""
Enumeration of reduced trinets (3-leaf semi-directed phylogenetic networks with a
single non-trivial 3-blob and no 2-sub-blobs) and their K2P Fourier parametrisation.

Conventions follow Brits, Holtgrefe, van Iersel & Martin (arXiv:2607.12919v3):
  Klein group G = Z2 x Z2 with A=(0,0), C=(0,1), G=(1,0), T=(1,1);
  Fourier parameters a_A = 1, a_C = a_T = X_e, a_G = Y_e for each edge e (K2P);
  network distribution = mixture over displayed trees (displayed-tree model).

A rooted binary network is stored as a dict:
  {'r': number of reticulations,
   'parents': {child: [parent,...]},   # leaves '1','2','3'; tree vertices 't*'; retics 'h*'; root 'rho'
  }
"""
import itertools, json
from functools import lru_cache
import networkx as nx

GROUP = {'A': (0, 0), 'C': (0, 1), 'G': (1, 0), 'T': (1, 1)}
NAME = {v: k for k, v in GROUP.items()}
LEAVES = ('1', '2', '3')


def gadd(*xs):
    s = (0, 0)
    for x in xs:
        g = GROUP[x]
        s = ((s[0] + g[0]) % 2, (s[1] + g[1]) % 2)
    return NAME[s]


# ---------------------------------------------------------------- enumeration
def rooted_networks(r):
    """All binary rooted phylogenetic networks on leaves 1,2,3 with r reticulations
    (labelled internal vertices; duplicates under relabelling are removed later)."""
    t = 3 + r - 2
    tree_v = [f't{i}' for i in range(t)]
    ret_v = [f'h{i}' for i in range(r)]
    internal = ['rho'] + tree_v + ret_v
    cap = {'rho': 2, **{v: 2 for v in tree_v}, **{v: 1 for v in ret_v}}
    out = []
    ret_choices = [list(itertools.combinations([p for p in internal if p != h], 2)) for h in ret_v]
    for rp in itertools.product(*ret_choices):
        used = {v: 0 for v in internal}
        ok = True
        for pair in rp:
            for p in pair:
                used[p] += 1
                if used[p] > cap[p]:
                    ok = False
        if not ok:
            continue
        for tp in itertools.product(*[[p for p in internal if p != tv] for tv in tree_v]):
            used2 = dict(used)
            ok = True
            for p in tp:
                used2[p] += 1
                if used2[p] > cap[p]:
                    ok = False
                    break
            if not ok:
                continue
            slots = []
            for v in internal:
                slots += [v] * (cap[v] - used2[v])
            if len(slots) != 3:
                continue
            for lp in set(itertools.permutations(slots)):
                parents = {}
                for h, pair in zip(ret_v, rp):
                    parents[h] = list(pair)
                for tv, p in zip(tree_v, tp):
                    parents[tv] = [p]
                for leaf, p in zip(LEAVES, lp):
                    parents[leaf] = [p]
                if valid_rooted(parents, internal):
                    out.append({'r': r, 'parents': parents})
    return out


def to_digraph(parents):
    D = nx.DiGraph()
    D.add_node('rho')
    for c, ps in parents.items():
        for p in ps:
            D.add_edge(p, c)
    return D


def valid_rooted(parents, internal):
    D = to_digraph(parents)
    if not nx.is_directed_acyclic_graph(D):
        return False
    # no parallel edges: a retic's two parents are distinct by construction; a tree vertex
    # cannot have two edges to the same child unless that child is a retic with both parents equal (excluded)
    # every vertex reachable from root
    if len(nx.descendants(D, 'rho')) != D.number_of_nodes() - 1:
        return False
    # root is the only vertex on all root-leaf paths (LSA condition)
    for v in internal:
        if v == 'rho':
            continue
        # if removing v disconnects all leaves from the root, v lies on all root-leaf paths
        D2 = D.copy()
        D2.remove_node(v)
        reach = nx.descendants(D2, 'rho')
        if not any(l in reach for l in LEAVES):
            return False
    return True


# ------------------------------------------------------- semi-directed form
def semidirected(parents):
    """Return the semi-directed network as a list of edges (u, v, directed_into_v_flag),
    after suppressing the root, identifying parallel edges and suppressing degree-2 vertices.
    Returns None if the root has two reticulation children (no consistent orientation)."""
    edges = []
    for c, ps in parents.items():
        for p in ps:
            edges.append([p, c, c.startswith('h')])  # directed iff child is a reticulation
    # suppress root
    re = [e for e in edges if e[0] == 'rho']
    assert len(re) == 2
    edges = [e for e in edges if e[0] != 'rho']
    (a, c1, d1), (b, c2, d2) = re
    if d1 and d2:
        return None
    if d1:
        edges.append([c2, c1, True])
    elif d2:
        edges.append([c1, c2, True])
    else:
        edges.append([c1, c2, False])
    changed = True
    while changed:
        changed = False
        # parallel edges
        seen = {}
        for i, (u, v, d) in enumerate(edges):
            key = frozenset((u, v))
            if key in seen:
                j = seen[key]
                eu, ev, ed = edges[j]
                # both must be directed into the same vertex (see text)
                assert d and ed and ev == v, ('unexpected parallel edges', edges[j], edges[i])
                edges[j] = [eu, ev, False]
                del edges[i]
                changed = True
                break
            seen[key] = i
        if changed:
            continue
        # degree-2 vertices
        deg = {}
        for u, v, d in edges:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        for w, dg in deg.items():
            if dg == 2 and w not in LEAVES:
                inc = [e for e in edges if w in e[:2]]
                (u1, v1, d1), (u2, v2, d2) = inc
                a = u1 if v1 == w else v1
                b = u2 if v2 == w else v2
                into_a = (v1 == a and d1)
                into_b = (v2 == b and d2)
                assert not (into_a and into_b)
                edges = [e for e in edges if w not in e[:2]]
                if into_b:
                    edges.append([a, b, True])
                elif into_a:
                    edges.append([b, a, True])
                else:
                    edges.append([a, b, False])
                changed = True
                break
    # reticulation status must be consistent: a vertex is a reticulation iff it has 2 incoming arcs
    return edges


def sd_graph(edges, labelled=True):
    """Encode a semi-directed network as a DiGraph for isomorphism testing."""
    D = nx.DiGraph()
    for u, v, d in edges:
        for x in (u, v):
            if x not in D:
                D.add_node(x, lab=(x if (labelled and x in LEAVES) else ('leaf' if x in LEAVES else 'int')))
        if d:
            D.add_edge(u, v, k='d')
        else:
            D.add_edge(u, v, k='u')
            D.add_edge(v, u, k='u')
    return D


def is_iso(g1, g2):
    return nx.is_isomorphic(g1, g2, node_match=lambda a, b: a['lab'] == b['lab'],
                            edge_match=lambda a, b: a['k'] == b['k'])


def analyse_sd(edges):
    """Return dict with structural info, or None if not a reduced trinet with a 3-blob."""
    U = nx.MultiGraph()
    for u, v, d in edges:
        U.add_edge(u, v)
    internal = [x for x in U.nodes if x not in LEAVES]
    # leaves attached to distinct internal vertices
    att = [list(U.neighbors(l))[0] for l in LEAVES]
    nret = sum(1 for x in internal if sum(1 for e in edges if e[1] == x and e[2]) == 2)
    # every reticulation must have exactly 2 incoming arcs and 1 other edge
    for x in internal:
        inc = sum(1 for e in edges if e[1] == x and e[2])
        if inc not in (0, 2):
            return None
    # single 3-blob: internal subgraph 2-edge-connected
    I = U.subgraph(internal).copy()
    if len(internal) == 1:
        return {'n_int': 1, 'ret': 0, 'tree': True}
    if len(set(att)) != 3 or not _multi_two_edge_connected(I):
        return None
    # no 2-sub-blobs: no connected W (|W|>=2, W != internal) with exactly two edges leaving W
    for k in range(2, len(internal)):
        for W in itertools.combinations(internal, k):
            Ws = set(W)
            if not nx.is_connected(U.subgraph(W)):
                continue
            leaving = sum(1 for u, v, d in edges if (u in Ws) != (v in Ws))
            if leaving == 2:
                return None
    return {'n_int': len(internal), 'ret': nret, 'tree': False}


def _multi_two_edge_connected(I):
    # 2-edge-connected for multigraph: removing any single edge keeps it connected
    if not nx.is_connected(I):
        return False
    for e in list(I.edges(keys=True)):
        J = I.copy()
        J.remove_edge(*e)
        if not nx.is_connected(J):
            return False
    return True


def reduced_trinets(r, labelled=True):
    """Classes (up to isomorphism) of reduced trinets with a 3-blob of strict level r.
    Returns list of dicts with a rooted representative and the semi-directed edges."""
    classes = []
    for net in rooted_networks(r):
        sd = semidirected(net['parents'])
        if sd is None:
            continue
        info = analyse_sd(sd)
        if info is None or info.get('tree') or info['ret'] != r:
            continue
        g = sd_graph(sd, labelled)
        for c in classes:
            if is_iso(c['g'], g):
                c['n_rootings'] += 1
                break
        else:
            classes.append({'g': g, 'sd': sd, 'rooted': net, 'n_rootings': 1})
    for c in classes:
        del c['g']
    return classes


# ------------------------------------------------------------ parametrisation
def displayed_trees(parents):
    """Yield (choice dict retic->index, list of tree edges (p,c), leaves-below dict per edge)."""
    rets = sorted([v for v in parents if v.startswith('h')])
    for sigma in itertools.product((0, 1), repeat=len(rets)):
        ch = dict(zip(rets, sigma))
        tedges = []
        for c, ps in parents.items():
            if c.startswith('h'):
                tedges.append((ps[ch[c]], c))
            else:
                tedges.append((ps[0], c))
        T = nx.DiGraph(tedges)
        below = {}
        for (p, c) in tedges:
            desc = nx.descendants(T, c) | {c}
            below[(p, c)] = frozenset(l for l in LEAVES if l in desc)
        yield ch, tedges, below


def edge_list(parents):
    return [(p, c) for c, ps in parents.items() for p in ps]
