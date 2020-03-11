import networkx as nx
import random
import sys, os
from operator import itemgetter
import satencoding
from itertools import repeat as _repeat, cycle
import subprocess
import signal
import math

# optional imports for debugging and plotting
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from time import time
from typing import List

RANDOM_SEED = 3
LOGGING = False
SAVEFIG = False
FIGCOUNTER = 0
DEEPEST_LEAF = False
PARTIAL_CONTRACTION = False
CONTRACTION_RATIO = 3
PARTIAL_CONTRACT_BY_DEPTH = False
MAXSAT = False

# optionally import wandb for logging purposes
try:
    import wandb
except ImportError:
    wandb = None
    pass


# utility functions

def always_true(x):
    return True


def first(obj):
    """
    return first element from object
    (also consumes it if obj is an iterator)
    """
    return next(iter(obj))


def pick(obj):
    """randomly pick an element from object"""
    return random.sample(obj, 1)[0]


def find_depth(dtree, root, return_deepest=False, ignore_weights=False):
    """
    find the depth of a tree (directed or undirected)
    fails if there are directed or undirected cycles
    also accounts for weights if they exist
    """
    maxdepth = 0
    deepest = None
    queue = [(root, 1)]
    while queue:
        node, depth = queue.pop(0)
        nodeweight = 0 if ignore_weights else dtree.nodes[node].get("weight", 0)
        if depth + nodeweight > maxdepth:
            maxdepth = depth + nodeweight
            deepest = node
        for child in dtree.neighbors(node):
            queue.append((child, depth+1))
    if return_deepest:
        return maxdepth, deepest
    else:
        return maxdepth


def find_root(digraph: nx.DiGraph):
    for i,d in digraph.in_degree:
        if d == 0:
            return i


def filter_ancestries(ancestries, graph: nx.Graph):
    return [(u,v) for u,v in ancestries if u in graph and v in graph]


def repeat(n, times=-1):
    if times >= 0:
        return _repeat(n, times)
    else:
        return _repeat(n)


# treedepth decomposition class
class TD(object):
    """class to hold a treedepth decomposition"""

    def __init__(self, tree, graph=None, root=None, depth=None):
        """
        tree is a heuristic decomposition as a directed tree
        graph is the original graph
        """
        self.tree = tree
        self.original_graph = graph
        self.root = root
        self.depth = depth
        self.leaves = set()
        if graph is not None:
            # represents current graph (with contractions etc)
            self.graph = graph.copy()
            self.contractions = []  # ordered list of contractions
            self.forced_ancestries = set()  # forced v parent of u (v->u)
        else:
            self.graph = None

    def __repr__(self):
        return f"<TD object: {list(self.tree.nodes)}>"

    def reset(self):
        """
        reset internal variables to prepare for a new wave of local improvements
        """
        self.graph = self.original_graph.copy()
        self.leaves = set()
        self.contractions = []
        self.forced_ancestries = set()

    def draw(self, attr=None, default=0, highlight=None):
        """
        draws a decomposition as a directed graph, with the root colored red
        """
        global FIGCOUNTER
        FIGCOUNTER += 1
        graph = self.tree
        pos = graphviz_layout(graph, prog="dot")  # or nx.spring_layout
        # draw all nodes
        nx.draw_networkx_nodes(graph, pos,
                               nodelist=set(graph.nodes))# - {self.root})
        # overdraw root if available
        if self.root is not None:
            "coloring root"
            nx.draw_networkx_nodes(graph, pos, node_color='r',
                                   nodelist=[self.root])
        if highlight is not None:
            nx.draw_networkx_nodes(graph, pos, node_color='g', nodelist=highlight)
        nx.draw_networkx_edges(graph, pos)
        # label vertices with index and weight
        if attr is not None:
            labels = dict((n, "{}:{}".format(n, d)) for n, d in graph.nodes.data(attr, default=default))
        else:
            labels = None
        nx.draw_networkx_labels(graph, pos, labels)
        if SAVEFIG:
            plt.savefig(f"figs/td_{FIGCOUNTER:03}.png", dpi=200)
        else:
            plt.show()
        plt.close()
        # todo[aesthetic]: draw graph edges in gray

    def write_to_file(self, filename):
        with open(filename, 'w') as f:
            if self.depth:
                f.write(f"{self.depth}\n")
            else:
                f.write(find_depth(self.depth, self.root))
            mapping = dict(enumerate(sorted(self.original_graph.nodes), start=1))
            invmapping = {v:k for k,v in mapping.items()}
            for i in range(1, len(self.original_graph)+1):
                if mapping[i] == self.root:
                    f.write("0\n")
                else:
                    parent = self.get_parent(mapping[i])
                    f.write(f"{invmapping[parent]}\n")

    def add_child(self, parent, child, **kwargs):
        """add a node to the decomposition, optionally as a child to parent"""
        self.leaves.clear()  # no longer consistent
        self.tree.add_node(child, **kwargs)
        if parent is not None:
            self.tree.add_edge(parent, child)

    def get_parent(self, child):
        if child == self.root:
            return None
        else:
            return first(self.tree.predecessors(child))  # todo[optional]: first -> only

    def copy(self):
        """return copy of decomposition, does not copy contractions"""
        tdcopy = TD(self.tree.copy(), self.original_graph, self.root, self.depth)
        tdcopy.annotate_leaves()
        # tdcopy.contractions = [(v, decomp.copy()) for v, decomp in self.contractions]
        # tdcopy.graph = None if self.graph is None else self.graph.copy()
        return tdcopy

    def annotate_leaves(self):
        """label each node of the decomposition if its a leaf or not"""
        self.leaves.clear()
        for vertex, degree in self.tree.out_degree:
            if degree == 0:
                self.leaves.add(vertex)
                # self.tree.nodes[vertex]["leaf"] = True
            # else:
            #     self.tree.nodes[vertex]["leaf"] = False

    def annotate_depth(self):
        """label each node of the decomposition with its depth"""
        assert self.root is not None, "root is None"
        queue = [(self.root, 1)]
        while queue:
            node, depth = queue.pop(0)
            self.tree.nodes[node]["depth"] = depth
            for child in self.tree.successors(node):
                queue.append((child, depth + 1))

    def annotate_subtree(self, debug=False):
        """label each node with the size of its subtree (prereq: annotate_leaves)"""
        assert self.root is not None, "root is None"
        data = self.tree.nodes
        # clear subtree data (except leaves)
        for node in self.tree.nodes:  # just for debugging todo[deploy]: remove later
            data[node]["subtree"] = -4
        for node in self.graph.nodes:
            if node in self.leaves:
                data[node]["subtree"] = 1
            else:
                data[node]["subtree"] = -2
        stack = [self.root]
        while stack:
            if debug: print("stack", stack)
            node = stack.pop()
            if debug: print("node", node)
            # if "subtree" in data[node]:  # node already computed
            #     if debug: print("skipping", node)
            #     continue
            node_subtree = 1
            remaining_children = []
            for child in self.tree.successors(node):
                child_subtree = data[child].get("subtree", -1)
                if child_subtree <= 0:
                    remaining_children.append(child)
                else:
                    node_subtree += child_subtree
            if remaining_children:
                if debug: print("children remaining", remaining_children)
                stack.append(node)
                stack.extend(remaining_children)
            else:
                if debug: print("no child remaining, subtree =", node_subtree)
                data[node]["subtree"] = node_subtree

    def annotate(self):
        """perform all the annotations"""
        self.annotate_leaves()
        self.annotate_depth()
        self.annotate_subtree()

    def get_descendants(self, root):
        """collect and return all descendants of a node (not just successors)"""
        desc = set()
        queue = [root]
        while queue:
            node = queue.pop(0)
            desc.add(node)
            for child in self.tree.successors(node):
                queue.append(child)
        return desc

    def get_leaves(self, deepest_first=DEEPEST_LEAF):
        leaves = list(self.leaves)
        if not deepest_first: return leaves
        data = self.tree.nodes
        leaves.sort(key=lambda l: data[l]["depth"]+data[l].get("weight", 0),
                    reverse=True)
        return leaves

    def get_subtree(self, root):
        descendants = self.get_descendants(root)
        subtree = self.tree.subgraph(descendants).copy()
        depth = find_depth(subtree, root)
        return TD(subtree, graph=None, root = root, depth = depth)

    def extract_subtree(self, budget, from_subset=None):
        """extract a subtree that fits the budget (prereq: annotate)"""
        data = self.tree.nodes
        feasible_root = None
        in_subset = always_true if from_subset is None else from_subset.__contains__
        for leaf in self.get_leaves():
            if not in_subset(leaf): continue
            node = leaf
            while True:
                if node == self.root:
                    return self.root
                parent = self.get_parent(node)
                if in_subset(parent) and data[parent]["subtree"] <= budget:
                    feasible_root = node = parent
                else:
                    if feasible_root is not None:
                        return feasible_root
                    else:
                        break
        return None

    def contract(self, local_root):
        """
        contract subtree rooted at local_root into single vertex,
        also update weight, ancestry and annotations accordingly
        """
        children = self.get_descendants(local_root) - {local_root}
        subtree = self.get_subtree(local_root)
        # update global list of contractions
        self.contractions.append((local_root, subtree))
        # set weight of local root
        current_weight = self.tree.nodes[local_root].get("weight", 0)
        assert current_weight == self.graph.nodes[local_root].get("weight", 0), f"weight attribute mismatch at {local_root}"
        new_weight = max(current_weight, subtree.depth - 1)  # for our case, current_weight is always 0
        self.tree.nodes[local_root]["weight"] = new_weight
        self.graph.nodes[local_root]["weight"] = new_weight
        # update ancestry
        potential_ancestors = nx.shortest_path(self.tree, self.root, local_root)
        potential_ancestors.pop()  # remove local_root
        for child in children:
            for potential_ancestor in potential_ancestors:
                if self.graph.has_edge(potential_ancestor, child):
                    self.forced_ancestries.add((potential_ancestor, local_root))
                    # todo[analyze,req] see if some forced ancestries can be removed/transferred directly
            self.graph = nx.contracted_nodes(self.graph, local_root, child, self_loops=False)
        # delete subtree
        self.tree.remove_nodes_from(children)
        # other book-keeping
        # todo[safe] could replace with single call to annotate
        self.leaves -= children
        self.leaves.add(local_root)
        subtree_size = len(subtree.tree) - 1
        for ancestor in potential_ancestors:
            self.tree.nodes[ancestor]["subtree"] -= subtree_size

    def push_up(self, node):
        assert node != self.root, "cannot push root up"
        parent = self.get_parent(node)
        node_weight = self.tree.nodes[node].get("weight", 0)
        prev_parent_weight = self.tree.nodes[parent].get("weight", 0)
        new_parent_weight = max(prev_parent_weight, node_weight + 1)
        self.tree.nodes[parent]["weight"] = new_parent_weight
        self.graph.nodes[parent]["weight"] = new_parent_weight
        potential_ancestors = nx.shortest_path(self.tree, self.root, parent)
        for potential_ancestor in potential_ancestors:
            if self.graph.has_edge(potential_ancestor, node):
                self.forced_ancestries.add((potential_ancestor, parent))
                # todo[analyze,req] see if some forced ancestries can be removed/transferred directly
        self.leaves.remove(node)
        self.graph = nx.contracted_nodes(self.graph, parent, node, self_loops=False)
        self.tree.remove_node(node)
        dtree = nx.DiGraph([(parent, node)])
        self.contractions.append((parent, TD(dtree, None, root=parent)))
        if self.tree.out_degree(parent) == 0: self.leaves.add(parent)
        for ancestor in potential_ancestors:
            self.tree.nodes[ancestor]["subtree"] -= 1

    def partial_contract_subroutine(self, contraction_size, local_nodes: set):
        """
        subroutine for partial contraction,
        returns True only if something was contracted
        """
        contraction_root = self.extract_subtree(contraction_size, local_nodes)
        if contraction_root is None:
            center, weights, labels = self.find_weighted_star(local_nodes)
            contraction_root = center  # contract obstruction
        descendants = self.get_descendants(contraction_root) - {contraction_root}
        if not descendants:
            print("#### no contraction made because no descendants")
            return False
        self.contract(contraction_root)
        local_nodes.difference_update(descendants)
        return True

    def partial_contract_by_size(self, local_root, contraction_size, target):
        """
        partially contract subtree rooted at local_root,
        until target ratio is achieved
        """
        subtree = self.get_subtree(local_root)
        local_nodes = set(subtree.tree.nodes)
        initial_size = len(local_nodes)

        if initial_size == 1:
            # push to parent
            print(f"#### single node contraction requested, pushing up")
            self.push_up(local_root)
            return

        changed = True
        while len(local_nodes) > target*initial_size and changed:
            changed = self.partial_contract_subroutine(contraction_size, local_nodes)

    def partial_contract_by_depth(self, local_root, contraction_size, target):
        """
        partially contract subtree rooted at local_root,
        until target decrease in depth is achieved
        """
        subtree = self.get_subtree(local_root)
        local_nodes = set(subtree.tree.nodes)
        initial_depth = find_depth(self.tree, local_root, ignore_weights=True)
        target_depth = max(initial_depth - target, 1)

        changed = True
        while find_depth(self.tree, local_root, ignore_weights=True) > target_depth and changed:
            changed = self.partial_contract_subroutine(contraction_size, local_nodes)

    def improve(self, local_root, new_decomps: List['TD']):
        """
        replaces subtree rooted at local_root with new_decomps
        """
        if local_root == self.root:  # replace entire decomp
            assert len(new_decomps) == 1, "Global decomp contains more than one component"
            decomp = new_decomps[0]
            self.tree = decomp.tree
            self.root = decomp.root
            self.depth = decomp.depth
        else:
            parent = self.get_parent(local_root)
            prev_descendants = self.get_descendants(local_root)
            new_descendants = set()
            for decomp in new_decomps:
                descendants = set(decomp.tree.nodes)
                assert descendants.issubset(prev_descendants), "improved decomposition contains foreign nodes:" \
                                                               f"{descendants-prev_descendants}"
                new_descendants.update(descendants)
                self.tree.remove_nodes_from(descendants)
                self.tree = nx.union(self.tree, decomp.tree)
                self.tree.add_edge(parent, decomp.root)
            if new_descendants != prev_descendants:
                print(f"#### missing nodes in improved decomposition: {new_descendants} vs {prev_descendants}")
        self.annotate()  # todo[opt] only update changed annotations

    def find_deepest_leaf(self):
        path_lengths = nx.single_source_dijkstra_path_length(self.tree, self.root)
        return max(path_lengths, key=path_lengths.get)

    def find_weighted_star(self, from_subset=None):
        #leaf = first(self.leaves)  # todo[exp]: try pick
        if from_subset is None:
            chosen_leaf = self.find_deepest_leaf()
        else:
            for leaf in self.get_leaves(deepest_first=True):
                if leaf in from_subset:
                    chosen_leaf = leaf
                    break
            else:
                raise RuntimeError("no leaf chosen while finding weighted star")
        parent = self.get_parent(chosen_leaf)
        assert parent is not None, "null parent while finding weighted star"
        descendants = self.get_descendants(parent)
        weights = [self.tree.nodes[parent].get("weight", 0)]
        labels = [parent]
        for descendant in descendants:
            if descendant == parent: continue
            weights.append(self.tree.nodes[descendant].get("weight", 0))
            labels.append(descendant)
        return parent, weights, labels

    def do_contractions(self, budget=4, debug=False, draw=False):
        """perform contractions given starting decomposition and budget"""
        # decomp.draw("subtree")
        # plt.show()
        while True:
            local_root = self.extract_subtree(budget)
            if local_root is None:  # obstructed by high-degree parent
                center, weights, labels = self.find_weighted_star()
                if debug: print("invoked weighted star:", center, weights)
                local_nodes = self.get_descendants(center)
                local_decomps = [linear_search(weights, labels, self.forced_ancestries)]
                local_root = center
            else:
                local_nodes = self.get_descendants(local_root)
                local_graph = self.graph.subgraph(local_nodes)
                known_depth = find_depth(self.tree, local_root)
                ancestries = filter_ancestries(self.forced_ancestries, local_graph)
                # further filtering will be handled component-wise by sat solver
                local_graph.graph["forced_ancestries"] = ancestries
                local_decomps = sat_solver(local_graph, known_depth)
            if draw:
                self.draw("weight", highlight=local_nodes)
                for local_decomp in local_decomps:
                    local_decomp.draw("weight")
            if debug:
                if len(local_decomps) > 1:
                    print("more than 1 decomp returned, disconnected components")
                for local_decomp in local_decomps:
                    print(f"old root:{local_root}\tnodes:{local_nodes}\ttd:{local_decomp.depth}\tnew root:{local_decomp.root}")
            if local_decomps:
                self.improve(local_root, local_decomps)
            else:  # no improvement found, just do contraction
                local_decomps = [self.get_subtree(local_root)]
            for local_decomp in local_decomps:
                if PARTIAL_CONTRACTION:
                    contraction_size = int(math.ceil(budget * CONTRACTION_RATIO))
                    if PARTIAL_CONTRACT_BY_DEPTH:
                        self.partial_contract_by_depth(local_decomp.root, contraction_size, 2)
                    else:
                        self.partial_contract_by_size(local_decomp.root, contraction_size, 0.5)
                else:
                    self.contract(local_decomp.root)
            self.annotate_subtree()  # maybe more annotations needed
            if self.root in local_nodes:  # reached root of heuristic decomposition
                break
        # print("treedepth after contraction:", self.tree.nodes[starting_root]["weight"]+1)

    def inflate_all(self, draw=False):
        """inflate decomposition to obtain full decomposition"""
        for node, local_decomp in reversed(self.contractions):
            if len(self.tree) == 1:
                self.tree = local_decomp.tree
            else:
                parent = self.get_parent(node)
                children = self.tree.successors(node)
                self.tree.remove_node(node)
                self.tree = nx.union(self.tree, local_decomp.tree)
                if parent is not None:
                    self.tree.add_edge(parent, local_decomp.root)
                for child in children:
                    self.tree.add_edge(node, child)
            if draw: self.draw(highlight=local_decomp.tree.nodes)


# upper bound heuristics

def simple_dfs(graph: nx.Graph, debug=False):
    """
    performs dfs from first node of a graph and returns 
    decomposition as a directed tree
    """
    source = first(graph.nodes)
    dfs_tree = nx.dfs_tree(graph, source)
    depth = find_depth(dfs_tree, source)
    return TD(dfs_tree, graph, source, depth)


def randomized_multiprobe_dfs(graph: nx.Graph, debug=False):
    """
    performs dfs from 10 randomly selected vertices and returns 
    the best decomposition among them as a directed tree
    """
    best_depth = 1e9
    best_decomp = None
    # todo[exp]: number of samples depends on size of graph
    num_samples = min(10, graph.number_of_nodes())
    sources = random.sample(graph.nodes, num_samples)
    for source in sources:
        dfs_tree = nx.dfs_tree(graph, source)
        depth = find_depth(dfs_tree, source)
        if depth < best_depth:
            best_depth = depth
            best_decomp = TD(dfs_tree, graph, source, depth)
    return best_decomp


def two_step_dfs(graph: nx.Graph, debug=False):
    """
    performs dfs from randomly selected vertex,
    then picks midpoint of longest path in this dfs tree 
    as the new source for dfs
    returns the best decomposition among them as a directed tree
    """
    source1 = pick(graph.nodes)
    dfs1 = nx.dfs_tree(graph, source1)
    depth1, deepest = find_depth(dfs1, source1, return_deepest=True)
    path = nx.shortest_path(dfs1, source1, deepest)
    source2 = path[len(path)//2]
    dfs2 = nx.dfs_tree(graph, source2)
    depth2 = find_depth(dfs2, source2)
    if debug: print(depth1, depth2)
    if depth1 < depth2:
        if debug: print("midpoint is worse")
        return TD(dfs1, graph, source1, depth1)
    else:
        return TD(dfs2, graph, source2, depth2)
    # todo[analyze]: find bad cases for two-step dfs


def lex_path(graph: nx.Graph, debug=False):
    """returns vertices in a path by lex order as a treedepth decomposition"""
    path = nx.DiGraph()
    nodes = sorted(graph.nodes)
    nx.add_path(path, nodes)
    return TD(path, graph, root=nodes[0], depth=len(nodes))


def random_path(graph: nx.Graph, debug=False):
    """returns vertices in a path in random order as a treedepth decomposition"""
    path = nx.DiGraph()
    nodes = sorted(graph.nodes)
    random.shuffle(nodes)
    nx.add_path(path, nodes)
    return TD(path, graph, root=nodes[0], depth=len(nodes))


HEURISTIC_FUNC = randomized_multiprobe_dfs
HEURISTIC_FUNCS = {"simple_dfs": simple_dfs,
                   "randomized_multiprobe_dfs": randomized_multiprobe_dfs,
                   "two_step_dfs": two_step_dfs,
                   "lex_path": lex_path,
                   "random_path": random_path}


# lower bound heuristic

def contraction2clique(graph: nx.Graph, debug=False):
    """
    contracts graph to clique and returns clique size
    (algorithm from [Gogate, Dechter 2003])
    """
    lowerbound = 0
    mindegree = lambda l: min(l, key=itemgetter(1))
    while len(graph) > 1:
        v, dv = mindegree(graph.degree)
        nv = list(graph.neighbors(v))
        lowerbound = max(lowerbound, dv)
        u, du = mindegree(graph.degree(nv))
        graph = nx.contracted_edge(graph, (u, v), self_loops=False)
        if debug: print("contracted", u, v)
    return lowerbound+1


# brute force algorithms

def brute_td(graph: nx.Graph, recdepth=0, debug=False):
    """computes exact treedepth of weighted graph by brute force approach"""
    if graph.number_of_nodes() == 1:
        node = first(graph.nodes)
        return 1 + graph.nodes.data("weight", default=0)[node]
    elif nx.is_connected(graph):
        nodes = graph.nodes
        mintd = 1e9
        minvert = None
        for node, weight in nodes.data("weight", default=0):
            if debug: print("  "*recdepth, "trying", node)
            newgraph = graph.subgraph(nodes - {node})
            td = max(weight, brute_td(newgraph, recdepth=recdepth+1, debug=debug))
            if td < mintd:
                mintd = td
                minvert = node
        if debug: print("  "*recdepth, "minvert:", minvert, "\tmintd:", mintd)
        return 1+mintd
    else:
        return max(brute_td(graph.subgraph(comp), recdepth=recdepth+1, debug=debug)
                   for comp in nx.connected_components(graph))


def get_brute_td(graph: nx.Graph, decomp=None, parent=None, debug=False):
    """
    computes the exact treedepth decomposition of a weighted graph
    by brute force approach
    """
    if decomp is None:
        decomp = TD(tree=nx.DiGraph(), root=parent, depth=0)
    else:
        decomp = decomp.copy()
    if graph.number_of_nodes() == 1:
        node = first(graph.nodes)
        weight = graph.nodes.data("weight", default=0)[node]
        decomp.add_child(parent, node, weight=weight)
        return 1 + weight, decomp
    elif nx.is_connected(graph):
        nodes = graph.nodes
        mintd = 1e9
        mindecomp = None
        minnode = None
        minnodeweight = None
        for node, weight in nodes.data("weight", default=0):
            # if debug: print("  "*recdepth, "trying", node)
            newgraph = graph.subgraph(nodes - {node})
            td, newdecomp = get_brute_td(newgraph, decomp, parent=node)
            td = max(weight, td)
            if td < mintd:
                mintd = td
                mindecomp = newdecomp
                minnode = node
                minnodeweight = weight
        # if debug: print("  "*recdepth, "minvert:", minvert, "\tmintd:", mintd)
        decomp = mindecomp
        decomp.add_child(parent, minnode, weight=minnodeweight)
        return 1+mintd, decomp
    else:
        maxtd = -1
        for comp in nx.connected_components(graph):
            td, decomp = get_brute_td(graph.subgraph(comp), decomp, parent=parent)
            if td > maxtd:
                maxtd = td
        return maxtd, decomp


# local improvement utilities

def get_weighted_star(weights):
    """
    constructs a weighted star with the given weights, the first element
    of weights being the weight of the center of the star
    """
    star = nx.star_graph(len(weights)-1)
    for node, weight in enumerate(weights):
        star.nodes[node]["weight"] = weight
    return star


def linear_search(weights, labels, ancestries, debug=False):
    """
    computes exact treedepth of a weighted star by a linear search algorithm
    #todo: [analyze] directly find point at which depth will be minimized
    """
    # combined = list(zip(weights, labels))
    center, center_label = weights[0], labels[0]
    forced_children = set()
    for parent, child in ancestries:
        assert child != center_label, "center of weighted star cannot be ancestor"
        if parent == center_label and child in labels:
            forced_children.add(child)
    forced_weight = 0
    unforced_combined = []
    weight_mapping = {center_label: center}
    for weight, label in zip(weights[1:], labels[1:]):
        weight_mapping[label] = weight
        if label in forced_children:
            forced_weight = max(forced_weight, weight)
        else:
            unforced_combined.append((weight, label))
    if unforced_combined:
        unforced_combined.sort(reverse=True)
        weights, labels = list(zip(*unforced_combined))
    else:
        weights = labels = ()
    mintd = 1e9
    minordering = None
    for i in range(len(weights)+1):
        remaining = max(forced_weight, max(weights[i:], default=0))
        # ordering = weights[:i] + (center,) + ((max(weights[i:]),) if weights[i:] else ())
        ordering = weights[:i] + (center, remaining)
        td = max(i+w for i, w in enumerate(ordering, start=1))
        if debug: print(td, weights[:i], center, remaining)
        if td < mintd:
            mintd = td
            minordering = labels[:i], labels[i:]
    if debug: print("minorder", minordering)
    # construct decomposition corresponding to the min ordering found
    mindecomp = nx.DiGraph()
    parents, rest = minordering
    decomp_root = center_label
    if parents:
        nx.add_path(mindecomp, parents)
        mindecomp.add_edge(parents[-1], center_label)
        decomp_root = parents[0]
    if rest:
        nx.add_star(mindecomp, (center_label,) + rest)
    nx.add_star(mindecomp, (center_label,) + tuple(forced_children))
    # replicate weights
    for label, weight in weight_mapping.items():
        mindecomp.nodes[label]["weight"] = weight
    return TD(mindecomp, root=decomp_root, depth=mintd)


num_sat_calls = 0
total_sat_calls = 0
def sat_solver(graph: nx.Graph, known_depth=-1):
    """forced_ancestries must be specified as a graph attribute"""
    global num_sat_calls
    num_sat_calls += 1
    ingraphsize = len(graph)
    if MAXSAT:
        decomptrees = satencoding.main_max(get_args(graph, known_depth))
    else:
        decomptrees = satencoding.main(get_args(graph, known_depth))
    decompsize = sum(map(len, decomptrees))
    if decompsize != ingraphsize:
        print(f"#### decomp mismatch {ingraphsize} {decompsize}")
    decomps = []
    for decomptree in decomptrees:
        droot = find_root(decomptree)
        depth = find_depth(decomptree, droot)
        decomps.append(TD(decomptree, graph, root=droot, depth=depth))
    return decomps


def local_improvement(given_decomp: TD, budget, debug=False, draw=False):
    """given_decomp contains a starting decomposition along with the graph"""
    decomp = given_decomp.copy()
    decomp.reset()
    decomp.annotate()
    decomp.do_contractions(budget=budget, debug=debug, draw=draw)
    decomp.inflate_all(draw=draw)
    decomp.depth = find_depth(decomp.tree, decomp.root)
    return decomp


def get_args(graph: nx.Graph, known_depth=-1):
    args = parser.parse_args(sys.argv[1:])
    args.preprocess = False  # turn off preprocessing
    args.instance = None
    args.graph = graph
    if known_depth >= 0:
        args.depth = known_depth
    return args


def draw_graph(g):
    nx.draw(g, with_labels=True)
    plt.show()


def no_data_graph(graph):
    new_graph = graph.__class__()
    new_graph.add_nodes_from(graph)
    new_graph.add_edges_from(graph.edges)
    return new_graph


def read_graph(filename: str):
    base, ext = os.path.splitext(filename)
    graph = None
    if ext == ".gr" or ext == ".edge":
        graph = satencoding.read_edge(filename)
    elif ext == ".gml":
        graph = nx.read_gml(filename, label=None)
    elif ext == ".gexf":
        graph = nx.read_gexf(filename)
    elif ext == ".graphml":
        graph = nx.read_graphml(filename)
    elif ext == ".dot":
        graph = nx.drawing.nx_agraph.read_dot(filename)
    elif ext == ".txt":
        graph = nx.read_edgelist(filename)
    else:
        raise ValueError("invalid graph file format")
    return no_data_graph(nx.Graph(graph))


def relabelled(g):
    return nx.convert_node_labels_to_integers(g)


def log_depth(filename, depth, total_time):
    api = wandb.Api()
    runs = api.runs("aditya95sriram/tdli-best", {"config.filename": filename})
    command = " ".join(sys.argv)
    githash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode()
    if len(runs) > 0:
        run = runs[0]
        previous_depth = run.summary["depth"]
        previous_time = run.summary.get("time", 1e6)
        if previous_depth > depth or (previous_depth == depth and previous_time > total_time):
            run.summary["depth"] = depth
            run.summary["command"] = command
            run.summary["githash"] = githash
            run.summary["time"] = total_time
            run.summary.update()
            print(f"###known bound({previous_depth}) >= current bound({depth})")
        else:
            print(f"###known bound({previous_depth}) < current bound({depth})")
    else:
        wandb.init(project="tdli-best", reinit=True)
        wandb.config.filename = filename
        wandb.log({"depth": depth, "command": command, "githash": githash,
                   "time": total_time})
        wandb.join()
        print("###registered first known bound", depth)


def solve_component(graph: nx.Graph, args, solution: 'Solution'):
    global num_sat_calls, total_sat_calls
    print("random state: {:x} {:x} {:x}".format(*random.getstate()[1][:3]))
    current_best = HEURISTIC_FUNC(graph)
    solution.update(current_best)
    print("random state: {:x} {:x} {:x}".format(*random.getstate()[1][:3]))
    single_budget = args.budget is not None
    if not single_budget:
        budget_range = range(5, 41, 5)
    else:
        budget_range = [args.budget]
    write_gr(graph, "cache.gr", comments=["command: " + " ".join(sys.argv),
                                          "githash: " + subprocess.check_output(
                                              ['git', 'rev-parse', '--short', 'HEAD']).strip().decode()])
    no_improvement_count = 0
    for budget_attempt in cycle(budget_range):
        for current_budget in repeat(budget_attempt, times=args.cap_tries):
            print("\ntrying budget", current_budget)
            num_sat_calls = 0
            try:
                new_decomp = local_improvement(current_best, current_budget, draw=args.draw_graphs)
            except SolverInterrupt:
                print("caught interrupt during subroutine")
                return current_best
            satencoding.verify_decomp(graph, new_decomp.tree, new_decomp.depth + 1, new_decomp.root)
            if new_decomp.depth < current_best.depth:
                solution.update(current_best)
                print(f"found improvement {current_best.depth}->{new_decomp.depth} with budget: {current_budget} [time: {time()-start_time:.2f}s]")
                current_best = new_decomp
                current_best.write_to_file("cache.tree")
                no_improvement_count = 0
                if LOGGING:
                    wandb.log({"best_depth": current_best.depth})
            else:
                print(f"no improvement ({current_best.depth}) with budget: {current_budget} [time: {time()-start_time:.2f}s]")
                no_improvement_count += 1
                break
            print(f"#sat calls: {num_sat_calls}")
            total_sat_calls += num_sat_calls
        if budget_attempt >= len(graph): break
        if no_improvement_count > 20:
            print("no improvement for 20 consecutive tries, quitting")
            break
    return current_best


def write_gr(graph: nx.Graph, filename: str, comments = None):
    with open(filename, 'w') as f:
        f.write("c generated by local_improvement.py\n")
        if comments is None: comments = []
        for comment in comments:
            f.write(f"c {comment}\n")
        f.write(f"p tdp {graph.number_of_nodes()} {graph.number_of_edges()}\n")
        mapping = {name: i for i, name in enumerate(sorted(graph.nodes), start=1)}
        for u, v in graph.edges:
            f.write(f"{mapping[u]} {mapping[v]}\n")


class Solution(object):
    def __init__(self):
        self.value = None

    def update(self, new_value):
        self.value = new_value


class SolverInterrupt(BaseException): pass


def term_handler(signum, frame):
    print("#### received signal", signum)
    raise SolverInterrupt


# SIGHUP SIGINT SIGUSR1 SIGUSR2 SIGTERM SIGSTOP
catch_signals = [1, 2, 10, 12, 15]

parser = satencoding.parser
parser.add_argument('-l', '--logging', action='store_true', help="Log run data to wandb")
parser.add_argument('-b', '--budget', type=int, help="budget for local instances")
parser.add_argument('-c', '--cap-tries', type=int, default=-1,
                    help="limit the number of attempts with the same budget")
parser.add_argument('-r', '--random-seed', type=int, default=3, help="random seed")
parser.add_argument('-j', '--just-sat', action='store_true',
                    help="one-shot SAT encoding, don't do local improvement")
parser.add_argument('--draw-graphs', action='store_true',
                    help="draw intermediate graphs for debugging purposes")
parser.add_argument('--heuristic', type=str, default="randomized_multiprobe_dfs",
                    help="heuristic function to be used for initial decomposition "
                         "[randomized_multiprobe_dfs*, simple_dfs, two_step_dfs, lex_path, random_path")
parser.add_argument('--deepest-leaf', action='store_true',
                    help="always pick deepest possible leaf for contraction")
parser.add_argument('-p', '--partial-contraction', type=float, default=0.2,
                    help="partial contraction ratio w.r.t budget, "
                         "do not contract entire subtree (target=budget/2)")
parser.add_argument('--partial-contract-by-depth', action='store_true',
                    help="partial contract by depth instead of size, reduce by 2 layers")
parser.add_argument('-m', '--max-sat', action='store_true',
                    help="use MaxSAT instead of linear search SAT")

if __name__ == '__main__':
    args = parser.parse_args()
    print("got args", args)
    LOGGING = args.logging
    RANDOM_SEED = args.random_seed
    SAVEFIG = args.draw_graphs
    HEURISTIC_FUNC = HEURISTIC_FUNCS[args.heuristic]
    DEEPEST_LEAF = args.deepest_leaf
    CONTRACTION_RATIO = args.partial_contraction
    if CONTRACTION_RATIO > 0:
        PARTIAL_CONTRACTION = True
    else:
        PARTIAL_CONTRACTION = False
    PARTIAL_CONTRACT_BY_DEPTH = args.partial_contract_by_depth
    MAXSAT = args.max_sat
    if args.instance is not None:
        filename = args.instance
    else:
        filename = "../pace-public/exact_005.gr"
    print("filename:", filename)
    for signalnum in catch_signals:
        signal.signal(signalnum, term_handler)
    start_time = time()
    input_graph = read_graph(filename)
    random.seed(RANDOM_SEED)
    satencoding.VIRTUALIZE = True
    current_depth = 0
    ncomps = 0
    for comp in nx.connected_components(input_graph):
        subgraph = input_graph.subgraph(comp)
        subtd = HEURISTIC_FUNC(subgraph)
        if args.just_sat:
            if args.depth >= 0:
                subtds = sat_solver(subgraph, min(args.depth, subtd.depth))
            else:
                subtds = sat_solver(subgraph, subtd.depth)
            current_depth = max(current_depth, max(td.depth for td in subtds))
        else:
            current_depth = max(current_depth, subtd.depth)
        ncomps += 1
    if args.just_sat:
        print("done (just sat), depth:", current_depth)
        sys.exit()
    basename = os.path.basename(filename)
    try:
        instance_type, instance_num = os.path.splitext(basename)[0].split("_")
        instance_num = int(instance_num)
    except ValueError:
        instance_type, instance_num = "other", 0
    if LOGGING:
        wandb.init(project="tdliexp1", tags=["cluster", instance_type],
                   reinit=True)
        wandb.config.instance_num = instance_num
        wandb.config.filename = basename
        wandb.config.seed = RANDOM_SEED
        wandb.config.n = len(input_graph)
        wandb.config.m = input_graph.number_of_edges()
        wandb.config.start_depth = current_depth
        wandb.config.timeout = args.timeout
        wandb.config.satstrat = "maxsat" if MAXSAT else "sat"
        wandb.config.partial_contraction = "depth" if PARTIAL_CONTRACT_BY_DEPTH else "size"
        wandb.config.contraction_ratio = CONTRACTION_RATIO
        wandb.config.job_id = os.environ.get("MY_JOB_ID", -1)
        wandb.config.task_id = os.environ.get("MY_TASK_ID", -1)
        wandb.log({"best_depth": current_depth})
        if args.budget is None:
            wandb.config.budget = -1
        else:
            wandb.config.budget = args.budget
    best_depth = 0
    icomp = 1
    solutions = []
    for comp in sorted(nx.connected_components(input_graph), key=len, reverse=True):
        print(f"working on comp {icomp}/{ncomps}, size:{len(comp)}")
        icomp += 1
        if len(comp) <= best_depth:
            print("skipping comp")
            continue
        subgraph = input_graph.subgraph(comp)
        random.seed(RANDOM_SEED)
        solution = Solution()
        try:
            subtd = solve_component(subgraph, args, solution)
        except SolverInterrupt:
            if ncomps > 1:
                print("interrupted during multi-component instance, results could be invalid")
            print("caught interrupt outside subroutine")
            subtd = solution.value
        solutions.append(subtd)
        best_depth = max(best_depth, subtd.depth)
    print("filename:", filename)
    logdata = {"best_depth": best_depth,
               "total_sat_calls": total_sat_calls, "time": time() - start_time}
    print("done, depth: {best_depth}/{start_depth}, n: {n}, m: {m}".format(n=len(input_graph),
                                                                           m=input_graph.number_of_edges(),
                                                                           start_depth=current_depth,
                                                                           **logdata))
    print("* total sat calls: {total_sat_calls}\ttotal time: {time:.3f}s".format(**logdata))
    write_gr(input_graph, "input.gr", comments=[f"file: {basename}",
                                                "command: " + " ".join(sys.argv),
                                                "githash: " + subprocess.check_output(
                                                    ['git', 'rev-parse', '--short', 'HEAD']).strip().decode()])
    for i, sol in enumerate(solutions, start=1):
        sol.write_to_file(f"sol{i}.tree")
    s = subprocess.check_output(["./verify", "input.gr", "sol1.tree"])
    print("\nExternal verification:", s.decode())
    if os.path.isfile("cache.gr"): os.remove("cache.gr")
    if os.path.isfile("cache.tree"): os.remove("cache.tree")
    if LOGGING:
        wandb.log({k:v for (k,v) in logdata.items() if k != 'best_depth'})
        wandb.join()
        log_depth(basename, best_depth, logdata["time"])
