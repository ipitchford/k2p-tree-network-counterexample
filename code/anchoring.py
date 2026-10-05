"""
Anchored trinets (sufficient condition for K2P tree-network distinguishability, Theorem C).

A reduced trinet N is *anchored at leaf x* if either
  (a) N has no non-trivial 3-blob (it reduces to the star tree), or
  (b) x is adjacent to a reticulation r of N, and for both incoming edges e of r the displayed network
      N - e, after full reduction (dangling parts removed, degree-2 vertices and parallel edges suppressed,
      2-sub-blobs suppressed), is anchored at x.
Networks are semi-directed edge lists [(u, v, directed_into_v), ...] with leaves '1','2','3'.
"""
import itertools
import networkx as nx

LEAVES = ('1', '2', '3')


def _deg(sd):
    d = {}
    for u, v, _ in sd:
        d[u] = d.get(u, 0) + 1; d[v] = d.get(v, 0) + 1
    return d


def cleanup(sd):
    sd = [list(e) for e in sd]
    changed = True
    while changed:
        changed = False
        deg = _deg(sd)
        # dangling internal vertices: degree <= 1, or every incident edge directed into the vertex
        for w in list(deg):
            if w in LEAVES:
                continue
            inc = [e for e in sd if w in e[:2]]
            if deg[w] <= 1 or all(e[1] == w and e[2] for e in inc):
                sd = [e for e in sd if w not in e[:2]]
                changed = True
                break
        if changed:
            continue
        # parallel edges
        seen = {}
        for i, (u, v, d) in enumerate(sd):
            key = frozenset((u, v))
            if key in seen:
                j = seen[key]
                eu, ev, ed = sd[j]
                if d and ed and ev == v:
                    sd[j] = [eu, ev, False]          # two arcs into the same vertex: merge, no longer reticulate
                elif not d and not ed:
                    pass
                else:
                    # an undirected and a directed parallel pair, or opposite arcs: merge keeping any arc
                    sd[j] = [u, v, d] if d else [eu, ev, ed]
                del sd[i]
                changed = True
                break
            seen[key] = i
        if changed:
            continue
        deg = _deg(sd)
        for w, dg in deg.items():
            if dg == 2 and w not in LEAVES:
                inc = [e for e in sd if w in e[:2]]
                (u1, v1, d1), (u2, v2, d2) = inc
                a = u1 if v1 == w else v1
                b = u2 if v2 == w else v2
                into_a = (v1 == a and d1); into_b = (v2 == b and d2)
                sd = [e for e in sd if w not in e[:2]]
                if into_b and not into_a:
                    sd.append([a, b, True])
                elif into_a and not into_b:
                    sd.append([b, a, True])
                else:
                    sd.append([a, b, False])
                changed = True
                break
    return [tuple(e) for e in sd]


def suppress_2subblobs(sd):
    while True:
        sd = cleanup(sd)
        U = nx.MultiGraph(); [U.add_edge(u, v) for u, v, _ in sd]
        internal = [x for x in U.nodes if x not in LEAVES]
        hit = None
        for k in range(2, len(internal)):
            for W in itertools.combinations(internal, k):
                Ws = set(W)
                if not nx.is_connected(U.subgraph(W)):
                    continue
                leaving = [e for e in sd if (e[0] in Ws) != (e[1] in Ws)]
                if len(leaving) == 2:
                    hit = (Ws, leaving); break
            if hit:
                break
        if not hit:
            return sd
        Ws, leaving = hit
        ends = []
        for u, v, d in leaving:
            out = v if u in Ws else u
            into_out = d and v == out
            ends.append((out, into_out))
        (a, ia), (b, ib) = ends
        sd = [e for e in sd if e[0] not in Ws and e[1] not in Ws]
        if ib and not ia:
            sd.append((a, b, True))
        elif ia and not ib:
            sd.append((b, a, True))
        else:
            sd.append((a, b, False))


def reduce_network(sd):
    return suppress_2subblobs(cleanup(sd))


def internal_vertices(sd):
    return sorted({x for e in sd for x in e[:2] if x not in LEAVES})


def is_star(sd):
    return len(internal_vertices(sd)) <= 1


def reticulations(sd):
    return [w for w in internal_vertices(sd) if sum(1 for e in sd if e[1] == w and e[2]) == 2]


def anchored(sd, x, depth=0, trace=None):
    sd = reduce_network(sd)
    if is_star(sd):
        return True
    nb = [e[0] if e[1] == x else e[1] for e in sd if x in e[:2]]
    r = nb[0]
    if r not in reticulations(sd):
        if trace is not None:
            trace.append(('  ' * depth) + f'leaf {x} not adjacent to a reticulation in {sd}')
        return False
    inc = [i for i, e in enumerate(sd) if e[1] == r and e[2]]
    for i in inc:
        sub = [e for j, e in enumerate(sd) if j != i]
        if not anchored(sub, x, depth + 1, trace):
            return False
    return True


if __name__ == '__main__':
    import json
    for lev in (1, 2):
        for i, c in enumerate(json.load(open(f'../data/trinets_level{lev}_unlabelled.json'))):
            sd = [tuple(e) for e in c['sd']]
            print(f'L{lev}-{i}: anchored at leaves', [x for x in LEAVES if anchored(sd, x)])
