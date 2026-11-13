"""Levels of concepts: one graph, nested partitions, additive statistics.

Leaves are the concepts of a fitted ``ConceptMixture``. They are generalized
bottom-up by the same description-length cost, without reassigning rows, so
every node is exactly the sum of its children. Levels are then read top-down:
the children of a node are the cut of its subtree that maximizes the pair
criterion measured against that node (docs/CONCEPT-GRAPH.md, G1-G5). Each level
is the prior of the next (G7). No count, depth, purity or similarity threshold.
"""
import numpy as np

from acan.concept_algebra import centre_variance, pair_mass, unbiased

KEYS = ('C', 'Cm', 'U', 'Q', 'N')


class ConceptGraph:
    def __init__(self, model, statistics, weights):
        self.model, self.weights = model, weights
        k = len(statistics['C'])
        self.leaves = k
        size = 2*k-1
        self.s = {key: np.zeros((size,)+statistics[key].shape[1:]) for key in KEYS}
        for key in KEYS:
            self.s[key][:k] = statistics[key]
        self.left, self.right = np.full(size, -1), np.full(size, -1)
        self._agglomerate(k)
        self.root = size-1
        self.children = {}
        self.depth = {self.root: 0}
        self._levels()

    # ---- bottom-up: generalization without reassignment ------------------------
    def _agglomerate(self, k):
        alive = list(range(k))                                  # node ids of the current concepts
        current = {key: self.s[key][:k].copy() for key in KEYS}
        cost = self.model._merge_costs(current, self.weights)
        for step in range(k-1):
            n = len(alive)
            flat = int(np.argmin(cost))
            i, j = sorted(divmod(flat, n))
            node = k+step
            self.left[node], self.right[node] = alive[i], alive[j]
            for key in KEYS:
                current[key][i] += current[key][j]
                self.s[key][node] = current[key][i]
                current[key] = np.delete(current[key], j, 0)
            alive[i] = node
            del alive[j]
            cost = np.delete(np.delete(cost, j, 0), j, 1)
            if n > 2:
                row = self.model._merge_costs(current, self.weights, rows=[i])[0]
                cost[i], cost[:, i] = row, row

    def members(self, node):
        """Leaf concepts below a node."""
        stack, found = [node], []
        while stack:
            item = stack.pop()
            if self.left[item] < 0:
                found.append(item)
            else:
                stack += [self.left[item], self.right[item]]
        return sorted(found)

    def _subtree(self, node):
        stack, found = [node], []
        while stack:
            item = stack.pop()
            found.append(item)
            if self.left[item] >= 0:
                stack += [self.left[item], self.right[item]]
        return np.sort(np.asarray(found))

    # ---- pair criterion against a reference node -------------------------------
    def relevance(self, nodes, reference):
        """Mean over attributes of 1 - (pair dispersion in the node)/(pair dispersion in the reference)."""
        wn, wc = self.weights
        encoder = self.model.encoder
        s = self.s
        total, weight = np.zeros(len(nodes)), 0.

        def variance(index):
            cm = s['Cm'][index]
            g = s['U'][index]/np.maximum(cm, 1e-12)
            return unbiased(np.maximum(s['Q'][index]/np.maximum(cm, 1e-12)-g*g, 0), cm, np)
        if len(wn):
            reference_variance = variance([reference])[0]
            known = reference_variance > 1e-12
            ratio = np.where(known, variance(nodes)/np.where(known, reference_variance, 1.), 1.)
            total += ((1-ratio)*wn).sum(1)
            weight += wn.sum()
        for name in encoder.categorical:
            block = encoder.block(name)

            def impurity(index):
                counts = s['N'][index][:, block]
                size = counts.sum(1)
                p = counts/np.maximum(size[:, None], 1e-12)
                return unbiased(1-(p*p).sum(1), size, np)
            reference_impurity = impurity([reference])[0]
            ratio = impurity(nodes)/reference_impurity if reference_impurity > 1e-12 else np.ones(len(nodes))
            total += (1-ratio)*wc[block][0]
            weight += wc[block][0]
        return total/max(weight, 1e-12)

    def _cut(self, reference):
        """Partition of the subtree maximizing sum pair_mass * relevance against ``reference`` (exact, by recursion on merges)."""
        nodes = self._subtree(reference)
        score = pair_mass(self.s['C'][nodes], np)*self.relevance(nodes, reference)
        position = {node: place for place, node in enumerate(nodes)}
        best, keep = score.copy(), np.ones(len(nodes), bool)
        for place, node in enumerate(nodes):                    # children have smaller ids than their parent
            if self.left[node] >= 0:
                split = best[position[self.left[node]]]+best[position[self.right[node]]]
                if split > score[place]+1e-12 or node == reference:
                    best[place], keep[place] = split, False
        if best[position[reference]] <= 1e-12:
            return [], 0.
        chosen, stack = [], [reference]
        while stack:
            node = stack.pop()
            if keep[position[node]] and node != reference:
                chosen.append(node)
            else:
                stack += [self.left[node], self.right[node]]
        return sorted(chosen), float(best[position[reference]])

    def _levels(self):
        pending = [self.root]
        while pending:
            node = pending.pop()
            if self.left[node] < 0:
                continue
            chosen, _ = self._cut(node)
            if not chosen:
                chosen = self.members(node)                      # nothing finer is supported: its leaves follow directly
            self.children[node] = chosen
            for child in chosen:
                self.depth[child] = self.depth[node]+1
                pending.append(child)

    # ---- every level is the prior of the next ----------------------------------
    def parameters(self):
        """Centre, variance and log-frequencies of the leaves, each shrunk to its parent level."""
        model, s = self.model, self.s
        encoder = model.encoder
        variance = {self.root: np.ones(s['U'].shape[1])}
        frequency = {self.root: model.root_p}
        order = sorted(self.depth, key=self.depth.get)
        parent = {child: node for node, kids in self.children.items() for child in kids}
        for node in order[1:]:
            up = parent[node]
            _, variance[node] = centre_variance(s['Cm'][node], s['U'][node], s['Q'][node], variance[up], model.kappa, np)
            size = np.zeros_like(s['N'][node])
            for name in encoder.categorical:
                block = encoder.block(name)
                size[block] = s['N'][node][block].sum()
            frequency[node] = (s['N'][node]+model.kappa*frequency[up])/(size+model.kappa)
        k = self.leaves
        g = s['U'][:k]/np.maximum(s['Cm'][:k], 1e-12)
        v = np.stack([variance[i] for i in range(k)])
        log_p = np.log(np.stack([frequency[i] for i in range(k)])) if encoder.width else np.zeros((k, 0))
        return g, v, log_p

    def levels(self):
        """Concepts at each depth (a node without finer levels stays itself) and their pair criterion against the root."""
        n = self.s['C'][self.root]
        result, front = [], [self.root]
        while True:
            following = [child for node in front for child in self.children.get(node, [node])]
            if following == front:
                break
            front = following
            nodes = np.asarray(front)
            value = (pair_mass(self.s['C'][nodes], np)*self.relevance(nodes, self.root)).sum()/max(n*(n-1), 1e-12)
            result.append(dict(depth=len(result)+1, concepts=len(front), pair_criterion=float(value), nodes=list(map(int, front))))
        return result
