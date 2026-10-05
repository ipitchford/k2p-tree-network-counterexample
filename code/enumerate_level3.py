"""
Enumerate reduced trinets with a strict level-3 3-blob (no 2-sub-blobs), up to isomorphism
(unlabelled leaves), by brute force over binary rooted networks with 3 reticulations.
Isomorphism classes are bucketed by a Weisfeiler-Lehman hash before exact isomorphism testing.
"""
import itertools, json, time, sys
import networkx as nx
from trinets import semidirected, analyse_sd, sd_graph, is_iso, LEAVES

r = int(sys.argv[1]) if len(sys.argv) > 1 else 3
t = 3 + r - 2
TV = [f't{i}' for i in range(t)]
RV = [f'h{i}' for i in range(r)]
INTERNAL = ['rho'] + TV + RV
CAP = {'rho': 2, **{v: 2 for v in TV}, **{v: 1 for v in RV}}
idx = {v: i for i, v in enumerate(INTERNAL + list(LEAVES))}

def acyclic_and_lsa(parents):
    D = nx.DiGraph()
    for c, ps in parents.items():
        for p in ps:
            D.add_edge(p, c)
    if not nx.is_directed_acyclic_graph(D):
        return False
    if len(nx.descendants(D, 'rho')) != D.number_of_nodes() - 1:
        return False
    for v in INTERNAL[1:]:
        D2 = D.copy(); D2.remove_node(v)
        reach = nx.descendants(D2, 'rho')
        if not any(l in reach for l in LEAVES):
            return False
    return True

t0 = time.time()
classes = {}          # wl-hash -> list of class dicts
n_valid = 0
ret_choices = [list(itertools.combinations([p for p in INTERNAL if p != h], 2)) for h in RV]
for rp in itertools.product(*ret_choices):
    used = dict.fromkeys(INTERNAL, 0)
    ok = True
    for pair in rp:
        for p in pair:
            used[p] += 1
            if used[p] > CAP[p]:
                ok = False
    if not ok:
        continue
    for tp in itertools.product(*[[p for p in INTERNAL if p != tv] for tv in TV]):
        u2 = dict(used); ok = True
        for p in tp:
            u2[p] += 1
            if u2[p] > CAP[p]:
                ok = False; break
        if not ok:
            continue
        slots = [v for v in INTERNAL for _ in range(CAP[v] - u2[v])]
        if len(slots) != 3:
            continue
        for lp in set(itertools.permutations(slots)):
            parents = {h: list(pair) for h, pair in zip(RV, rp)}
            parents.update({tv: [p] for tv, p in zip(TV, tp)})
            parents.update({l: [p] for l, p in zip(LEAVES, lp)})
            if not acyclic_and_lsa(parents):
                continue
            n_valid += 1
            sd = semidirected(parents)
            if sd is None:
                continue
            g = sd_graph(sd, labelled=False)
            h = nx.weisfeiler_lehman_graph_hash(g, node_attr='lab', edge_attr='k')
            bucket = classes.setdefault(h, [])
            for c in bucket:
                if is_iso(c['g'], g):
                    c['n'] += 1
                    break
            else:
                info = analyse_sd(sd)
                bucket.append({'g': g, 'sd': sd, 'parents': parents, 'n': 1, 'info': info})
print(f'valid rooted networks: {n_valid}; time {time.time()-t0:.0f}s')
reduced = [c for b in classes.values() for c in b if c['info'] and not c['info'].get('tree') and c['info']['ret'] == r]
print(f'semi-directed classes: {sum(len(b) for b in classes.values())}; reduced strict level-{r} trinet shapes: {len(reduced)}')
json.dump([{'sd': c['sd'], 'rooted': {'r': r, 'parents': c['parents']}, 'n_rootings': c['n']} for c in reduced],
          open(f'../data/trinets_level{r}_unlabelled.json', 'w'), indent=1)
