import networkx as nx
import random
import sys
from operator import itemgetter
import satencoding

# optional imports for debugging and plotting
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from time import time

RANDOM_SEED = 3
LOGGING = False
SAVEFIG = False
FIGCOUNTER = 0

if LOGGING: import wandb
# utility functions

def first(obj):
    """
    return first element from object
    (also consumes it if obj is an iterator)
    """
    return next(iter(obj))


def pick(obj):
    """randomly pick an element from object"""
    return random.sample(obj, 1)[0]


def find_depth(dtree, root, return_deepest=False):
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
        nodeweight = dtree.nodes[node].get("weight", 0)
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

    def add_child(self, parent, child, **kwargs):
        """add a node to the decomposition, optionally as a child to parent"""
        self.leaves.clear()  # no longer consistent
        self.tree.add_node(child, **kwargs)
        if parent is not None:
            self.tree.add_edge(parent, child)

    def get_parent(self, child):
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

    def extract_subtree(self, budget):
        """extract a subtree that fits the budget (prereq: annotate)"""
        data = self.tree.nodes
        feasible_root = None
        for leaf in self.leaves:
            node = leaf
            while True:
                if node == self.root:
                    return self.root
                parent = self.get_parent(node)
                if data[parent]["subtree"] <= budget:
                    feasible_root = node = parent
                else:
                    if feasible_root is not None:
                        return feasible_root
                    else:
                        break
        return None

    def contract(self, local_decomp: 'TD', descendants):
        """
        contract subtree rooted at root into single vertex with weight
        desc is the set of descendants of root
        """
        # add root->subtree into self.contractions
        root = local_decomp.root
        assert root is not None, "root of local decomp is none"
        # check if current root is in local instance
        if self.root in descendants:
            self.root = root  # update with new (possibly same) root
        self.contractions.append((root, local_decomp))
        # mark root as leaf
        self.leaves -= descendants
        self.leaves.add(root)
        # set weight of root
        weight = local_decomp.depth - 1  # exclusive weight convention
        self.tree.nodes[root]["weight"] = weight
        self.graph.nodes[root]["weight"] = weight
        # update self.graph with contractions
        for child in descendants - {root}:
            outside_neighbors = set(self.graph.neighbors(child)) - descendants
            for out_nbr in outside_neighbors:
                self.forced_ancestries.add((out_nbr, root))
                # todo[analyze,req] see if some forced ancestries can be removed/transferred directly
            self.graph = nx.contracted_nodes(self.graph, root, child, self_loops=False)
            self.tree = nx.contracted_nodes(self.tree, root, child, self_loops=False)

    def find_deepest_leaf(self):
        path_lengths = nx.single_source_dijkstra_path_length(self.tree, self.root)
        return max(path_lengths, key=path_lengths.get)

    def find_weighted_star(self):
        #leaf = first(self.leaves)  # todo[exp]: try pick
        leaf = self.find_deepest_leaf()
        parent = self.get_parent(leaf)
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
                local_decomp = linear_search(weights, labels, self.forced_ancestries)
                local_root = center
            else:
                local_nodes = self.get_descendants(local_root)
                local_graph = self.graph.subgraph(local_nodes)
                ancestries = filter_ancestries(self.forced_ancestries, local_graph)
                local_graph.graph["forced_ancestries"] = ancestries
                local_decomp = sat_solver(local_graph)
            if draw:
                self.draw("weight", highlight=local_nodes)
                local_decomp.draw("weight")
            if debug:
                print(f"old root:{local_root}\tnodes:{local_nodes}\ttd:{local_decomp.depth}\tnew root:{local_decomp.root}")
            # todo[analyze]: do you need local nodes or is it in decomp
            self.contract(local_decomp, local_nodes)
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
                parent = self.get_parent(node) if node != self.root else None
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
    for weight, label in zip(weights[1:], labels[1:]):
        if label in forced_children:
            forced_weight = max(forced_weight, weight)
        else:
            unforced_combined.append((weight, label))
    unforced_combined.sort(reverse=True)
    weights, labels = list(zip(*unforced_combined))
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
    return TD(mindecomp, root=decomp_root, depth=mintd)

num_sat_calls = 0
total_sat_calls = 0
def sat_solver(graph: nx.Graph):
    """forced_ancestries must be specified as a graph attribute"""
    global num_sat_calls
    num_sat_calls += 1
    ingraphsize = len(graph)
    lb, ub, decomptree = satencoding.main(get_args(graph))
    decompsize = len(decomptree)
    assert decompsize == ingraphsize, f"decomp mismatch {ingraphsize} {decompsize}"
    droot = find_root(decomptree)
    depth = find_depth(decomptree, droot)
    return TD(decomptree, graph, root=droot, depth=depth)


def local_improvement(given_decomp: TD, budget, debug=False, draw=False):
    """given_decomp contains a starting decomposition along with the graph"""
    decomp = given_decomp.copy()
    decomp.reset()
    decomp.annotate()
    decomp.do_contractions(budget=budget, debug=debug, draw=draw)
    decomp.inflate_all(draw=draw)
    decomp.depth = find_depth(decomp.tree, decomp.root)
    return decomp


def get_args(graph: nx.Graph):
    args = satencoding.parser.parse_args(["--timeout", "2", "--no-preprocess"])
    args.graph = graph
    return args


def draw_graph(g):
    nx.draw(g, with_labels=True)
    plt.show()


def relabelled(g):
    return nx.convert_node_labels_to_integers(g)


# all cases
# seed=3, two_step_dfs, dim=3, budget=3: 6->5[5]
# seed=3, two_step_dfs, dim=5, budget=3: 25->16[9]
# wrong cases
# *seed=3, simple_dfs, dim=3, budget=3: edge not covered; NOW WORKING, 9->6[5]
# *seed=3, two_step_dfs, dim=4, budget=3: edge not covered; NOW WORKING, 12->9[7]
# *seed=3, two_step_dfs, dim=4, budget=4: invalid decompostition; NOW WORKING, 12->9[7]
# *seed=3, two_step_dfs, dim=6, budget=5: edge not covered; NOW WORKING, 33-24[11]
# *seed=3, randomize_multiprobe, budget=[5:31:5], pace-public#005: edge not covered; NOW WORKING
# *seed=3, randomize_multiprobe, budget=[5:31:5], pace-public#079: invalid literal UNSAT; NOW WORKING
# *seed=3, randomize_multiprobe, budget=[5:31:5], pace-public#081: KeyError, node not in DiGraph; NOW WORKING
# seed=3, randomize_multiprobe, budget=[5:31:5], pace-public#135: invalid literal UNSAT; edge not covered
# seed=3, randomize_multiprobe, budget=[5:31:5], pace-public#179: KeyError
# seed=3, randomize_multiprobe, budget=[5:31:5], pace-public#197: KeyError
# seed=3, randomize_multiprobe, budget=[5:31:5], pace-public#199: edge not covered
if __name__ == '__main__':
    if len(sys.argv) > 1:
        instance_num = int(sys.argv[1])
    else:
        instance_num = 135
    filename = f"../pace-public/exact_{instance_num:03}.gr"
    print("filename:", filename)
    start_time = time()
    # grid_dim = 6
    # _g = nx.grid_2d_graph(grid_dim, grid_dim)
    # _g = nx.convert_node_labels_to_integers(_g)
    # _g = nx.Graph(satencoding.read_edge("../../treedepth-sat/inputs/famous/B10Cage.edge"))
    input_graph = relabelled(nx.Graph(satencoding.read_edge(filename)))
    random.seed(RANDOM_SEED)
    satencoding.VIRTUALIZE = True
    current_best = randomized_multiprobe_dfs(input_graph)
    heuristic_depth = current_best.depth
    input_size = len(input_graph)
    if LOGGING:
        wandb.init(project="tdli")
        wandb.config.seed = RANDOM_SEED
    for current_budget in range(5, 31, 5):
        print("\ntrying budget", current_budget)
        total_sat_calls += num_sat_calls
        num_sat_calls = 0
        new_decomp = local_improvement(current_best, current_budget, draw=False)
        satencoding.verify_decomp(input_graph, new_decomp.tree, new_decomp.depth + 1, new_decomp.root)
        if new_decomp.depth < current_best.depth:
            print(f"\nfound improvement {current_best.depth}->{new_decomp.depth} with budget: {current_budget}")
            current_best = new_decomp
        else:
            print(f"\nno improvement with budget: {current_budget}")
        print(f"#sat calls: {num_sat_calls}")
        if current_budget >= input_size: break
    print("filename:", filename)
    logdata = {"filename": filename, "best_depth": current_best.depth, "n": len(input_graph),
               "m": input_graph.number_of_edges(), "total_sat_calls": total_sat_calls,
               "time": time() - start_time, "final_budget": current_budget,
               "start_depth": heuristic_depth}
    print("done, best depth: {best_depth}, n: {n}, m: {m}, final budget: {final_budget}".format(**logdata))
    print("* total sat calls: {total_sat_calls}\ttotal time: {time:.3f}s".format(**logdata))
    if LOGGING: wandb.log(logdata)

    # if len(input_graph) <= 50:
    #     print("\nattempting sat solver on input_graph")
    #     decomp = sat_solver(input_graph)
    #     print(f"sat solution depth: {decomp.depth}")

    # print("treedepth:", brute_td(_g))
    # print("clique size:", contraction2clique(_g))
    # _decomp2 = local_improvement(_g, 20, two_step_dfs(_g), draw=False)
    # _decomp2.draw()
    # print(f"new depth: {_decomp2.depth}\t #sat calls: {num_sat_calls}")
    # satencoding.verify_decomp(_g, _decomp2.tree, _decomp2.depth+1, _decomp2.root)
    # _decomp3 = TD(_decomp2.tree, _g, _decomp2.root, _decomp2.depth)
    # _decomp2.reset()
    # _decomp4 = local_improvement(_g, 3, _decomp3, draw=False)
    # print(f"new depth: {_decomp4.depth}\t #sat calls: {num_sat_calls}")
    # satencoding.verify_decomp(_g, _decomp4.tree, _decomp4.depth+1, _decomp4.root)
    # _g2 = nx.cycle_graph(3)
    # _g2.graph["forced_ancestries"] = [(2,0), (1,0)]
    # _g2.nodes[0]["weight"] = 1
    #
    # # sat encoding and solving
    # import os, subprocess
    # i, temp, instance = 5, "./tmp", "trial"
    # encoding = satencoding.generate_encoding(_g2, i)
    # print(encoding)
    # cnf = os.path.join(temp, instance + '_' + str(i) + ".cnf")
    # with open(cnf, 'w') as ofile:
    #     ofile.write(encoding)
    # sol = os.path.join(temp, instance + '_' + str(i) + ".sol")
    # cmd = ["glucose", cnf, sol]
    # # print cmd
    # p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # output, err = p.communicate()
    # rc = p.returncode
    # print("retcode", rc)
    # if rc == 10:
    #     sol_file = os.path.join(temp, instance + '_' + str(i) + ".sol")
    #     decomp = satencoding.decode_output(sol_file, _g2, i)