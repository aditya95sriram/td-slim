# coding=utf-8
import argparse
import os
import sys
import time
import networkx as nx
import subprocess
from operator import itemgetter

from networkx.drawing.nx_agraph import *
import matplotlib.pyplot as plt
import signal

VIRTUALIZE = False

def apex_vertices(g):
    buff = 0
    delete_vertices = list()
    for u, degree in g.degree():
        if degree == g.number_of_nodes() - 1:
            delete_vertices.append(u)
    g.remove_nodes_from(delete_vertices)
    buff += len(delete_vertices)
    nx.convert_node_labels_to_integers(g, first_label=0)
    return g, buff


def degree_one_reduction(g):
    """
    Removes all but one degree one neighbours of one vertex
    :returns g: reduced graph
    :type g: networkx graph
    """
    nodes = set()
    for u in g.nodes():
        deg = 0
        for v in g.neighbors(u):
            if g.degree(v) == 1:
                if deg == 0:
                    deg = 1
                else:
                    nodes = nodes.union({v})
    g.remove_nodes_from(list(nodes))
    g = nx.convert_node_labels_to_integers(g, first_label=0)
    return g


def read_edge(filename):
    with open(filename, 'r') as in_file:
        edge = in_file.read()
    edge = edge.replace('e ', '')
    edge = edge.split('\n')
    while edge[0][0] != 'p':
        edge.pop(0)
    attr = edge.pop(0)
    attr = attr.split()
    attr = int(attr.pop())
    while len(edge) > attr:
        edge.pop()
    int_edge = list()
    for e in edge:
        eu, ev = e.split()
        int_edge.append([int(eu), int(ev)])
    if int_edge[len(int_edge) - 1] == []:
        int_edge.pop()
    return int_edge


def make_vars(g, width):
    nv = g.number_of_nodes()
    p = [[[0 for i in range(width)] for j in range(nv)] for k in range(nv)]
    nvar = 1
    for u in range(nv):
        for v in range(u, nv):
            for i in range(width):
                p[u][v][i] = nvar
                nvar += 1
    return p, nvar - 1


def generate_encoding(g, reqwidth):
    nv = g.number_of_nodes()
    if VIRTUALIZE and reqwidth > nv:
        width = nv+1
        virtualizing = True
        delta = reqwidth - width
    else:
        width = reqwidth
        virtualizing = False
        delta = -1  # not needed
    s, nvar = make_vars(g, width)
    encoding = ''
    nclauses = 0
    for u in range(nv):
        for v in range(u, nv):
            encoding += '%i 0\n' % (s[u][v][width - 1])
            nclauses += 1
            encoding += '-%i 0\n' % (s[u][v][0])
            nclauses += 1
    for u in range(nv):
        for v in range(u, nv):
            for i in range(1, width):
                encoding += '-%i %i 0\n' % (s[u][v][i - 1], s[u][v][i])
                nclauses += 1
    for u in range(nv):
        for v in range(u + 1, nv):
            for w in range(v + 1, nv):
                for i in range(width):
                    encoding += '-%i -%i %i 0\n' % (s[u][v][i], s[u][w][i], s[v][w][i])
                    nclauses += 1
                    encoding += '-%i -%i %i 0\n' % (s[u][v][i], s[v][w][i], s[u][w][i])
                    nclauses += 1
                    encoding += '-%i -%i %i 0\n' % (s[u][w][i], s[v][w][i], s[u][v][i])
                    nclauses += 1
    for u in range(nv):
        for v in range(u + 1, nv):
            for i in range(width):
                encoding += '-%i %i 0\n' % (s[u][v][i], s[u][u][i])
                encoding += '-%i %i 0\n' % (s[u][v][i], s[v][v][i])
                nclauses += 2
    for u in range(nv):
        for v in range(u + 1, nv):
            for i in range(1, width):
                encoding += '-%i %i %i 0\n' % (s[u][v][i], s[u][u][i - 1], s[v][v][i - 1])
                nclauses += 1
    for e in g.edges():
        u = min(e)
        v = max(e)
        if u == 3 and v == 4:
            for i in range(1, width):
                encoding += 'c new\n'
                encoding += '-%i -%i %i 0\n' % (s[u][v][i], s[v][v][i - 1], s[u][v][i - 1])
                nclauses += 1
        for i in range(1, width):
            encoding += '-%i %i -%i %i 0\n' % (s[u][u][i], s[u][u][i - 1], s[v][v][i], s[u][v][i])
            nclauses += 1
            encoding += '-%i %i -%i %i 0\n' % (s[u][u][i], s[v][v][i - 1], s[v][v][i], s[u][v][i])
            nclauses += 1
    # weight encoding constraints
    weight_encoding = "c weight encoding\n"
    weight_nclauses = 0
    for u, weight in g.nodes.data("weight"):
        if weight:
            #weight = d["weight"] + 1
            #if weight <= 1: continue
            if not virtualizing:
                if weight >= width:  # NO instance
                    weight_encoding += "c UNSAT\n%i 0\n-%i 0\n" % (1, 1)  # UNSAT
                    weight_nclauses += 2
                else:
                    # constraint: not s(u,u,w)
                    weight_encoding += "-%i 0\n" % (s[u][u][weight])
                    weight_nclauses += 1
            else:
                if weight <= delta:
                    pass  # no constraints needed
                else:
                    # constraint: not s(u,u,w-(D-n))
                    weight_encoding += "-%i 0\n" % (s[u][u][weight - delta])
                    weight_nclauses += 1
            # todo: implement flipped indexing (-s[u][u][width - weight])
            # weight_encoding += '%i 0\n' % (s[u][u][weight])  # constraint: not s(u,u,D-w)
            # weight_nclauses += 1
            # for v in range(nv):
            #     if u == v: continue
            #     weight_encoding += '-%i 0\n' % (s[min(u, v)][max(u, v)][weight])
            #     weight_nclauses += 1
    # forced ancestry encoding constraints
    ancestry_encoding = "c ancestry encoding\n"
    ancestry_nclauses = 0
    forced_ancestries = g.graph.get("forced_ancestries", [])
    for parent, child in forced_ancestries:
        for i in range(2, width+1):
            ancestry_encoding += "-%i %i 0\n" % (s[parent][parent][i-1], s[child][child][i-1])
            ancestry_nclauses += 1
    preamble = 'p cnf %i %i\n' % (nvar, nclauses + weight_nclauses + ancestry_nclauses)
    comment = 'c %s\n' % ("\nc ".join(map(str, s))); comment += "c virtualizing: %d\n" % virtualizing; encoding += comment
    return preamble + encoding + weight_encoding + ancestry_encoding


def decode_output(sol, g, reqwidth, return_decomp=False):
    nv = g.number_of_nodes()
    if VIRTUALIZE and reqwidth > nv:
        width = nv+1
    else:
        width = reqwidth
    with open(sol, 'r') as out_file:
        out = out_file.read()
    out = out.split('\n')
    out = out[0]
    out = out.split(' ')
    out = list(map(int, out))
    out.pop()
    ne = g.number_of_edges()
    s, nvar = make_vars(g, width)
    components = list()
    for i in range(width - 1, 0, -1):
        level = list()
        for u in range(nv):
            ver = list()
            for v in range(u, nv):
                if out[s[u][v][i] - 1] > 0:
                    ver.append(v)
            do_not_add = 0
            for v in level:
                if set(ver).issubset(set(v)):
                    do_not_add = 1
            if do_not_add == 0:
                level.append(ver)
        components.append(level)
    for i in components:
        sys.stderr.write(str(i) + '\n')
    sys.stderr.write('\n' + "*" * 10 + '\n')
    decomp = nx.DiGraph()
    root = list()
    level_i = list()
    for i in range(width - 1, 0, -1):
        level = list()
        # sys.stderr.write('\n'+'*'*10+'\n')
        for u in range(nv):
            if out[s[u][u][i] - 1] > 0 and out[s[u][u][i - 1] - 1] < 0:
                edge_add = False
                if i == width - 1:
                    root.append(u)
                decomp.add_node(u, level=i)
                # sys.stderr.write("%i "%u)
                level.append(u)
                if level_i != []:
                    for v in level_i[len(level_i) - 1]:
                        if out[s[min(u, v)][max(u, v)][i + 1] - 1] > 0:
                            decomp.add_edge(v, u)
                            edge_add = True
                if not edge_add:
                    if level_i != []:
                        level_u = 1e9 #g.number_of_nodes()
                        v_u = -1
                        for v in decomp.nodes(data=True):
                            if v[0] == u:
                                continue
                            if out[s[min(u, v[0])][max(u, v[0])][v[1]['level']] - 1] > 0:
                                if level_u > v[1]['level']:
                                    level_u = v[1]['level']
                                    v_u = v[0]
                        decomp.add_edge(v_u, u)
        level_i.append(level)
        # print level
    # show_graph(decomp,1)
    verify_decomp(g=g, s=decomp, width=width, root=root)
    if return_decomp: return decomp


def verify_decomp(g, s, width, root):
    sys.stderr.write("\nValidating tree depth decomposition\n")
    sys.stderr.flush()
    # print g.edges()
    for e in g.edges():
        try:
            nx.shortest_path(s, e[0], e[1])
        except:
            try:
                nx.shortest_path(s, e[1], e[0])
            except:
                raise Exception("Edge %i %i not covered\n" % (e[0], e[1]))
    for v, d in g.degree():
        count = 0
        if d != 1:
            continue
        for i in root:
            try:
                if len(nx.shortest_path(s, i, v)) - 1 > width:
                    raise ValueError("depth of tree more than width\n")
            except:
                count += 1
                if count == len(root):
                    raise Exception("No root found for %i\n" % v)
                continue
    sys.stderr.write("Valid treedepth decomp\n")
    sys.stderr.flush()


def show_graph(graph, layout=1, nolabel=0):
    """ show graph
    layout 1:graphviz,
    2:circular,
    3:spring,
    4:spectral,
    5: random,
    6: shell
    """

    m = graph.copy()
    if layout == 1:
        pos = graphviz_layout(m)
    elif layout == 2:
        pos = nx.circular_layout(m)
    elif layout == 3:
        pos = nx.spring_layout(m)
    elif layout == 4:
        pos = nx.spectral_layout(m)
    elif layout == 5:
        pos = nx.random_layout(m)
    elif layout == 6:
        pos = nx.shell_layout(m)
    else:
        pos = nx.spring_layout(m)
    if not nolabel:
        nx.draw_networkx_edge_labels(m, pos)
    nx.draw_networkx_labels(m, pos)
    nx.draw_networkx_nodes(m, pos)
    # write_dot(m, "m1.dot")
    # os.system("dot -Tps m1.dot -o m1.ps")
    nx.draw(m, pos)
    plt.show()


class Timer(object):
    def __init__(self, time_list=None):
        self.time_list = time_list

    def __enter__(self):
        self.start = time.time()
        self.end = self.duration = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.duration = self.end - self.start
        if self.time_list is not None:
            self.time_list.append(self.duration)
        if exc_val is None:
            self.err = None
        else:
            self.err = (exc_type, exc_val, exc_tb)
            # print >> sys.stderr, "\ntimed block terminated abruptly after", self.duration, "seconds"
            # print >> sys.stderr, self.err
            print("\ntimed block terminated abruptly after", self.duration, "seconds", file=sys.stderr)
            print(self.err, file=sys.stderr)


def solve_component(g, cli_args):
    lb = 0
    ub = 0
    to = False
    encoding_times = list()
    solving_times = list()
    n = g.number_of_nodes()
    if n <= 1:
        return n, n, n, to, encoding_times, solving_times
    temp = os.path.abspath(cli_args.temp)
    instance = cli_args.instance
    # looprange = range(g.number_of_nodes() + 2, 1, -1)  # original
    maxweight = max(map(itemgetter(1), g.nodes.data("weight", default=0)))
    looprange = range(g.number_of_nodes() + maxweight + 1, maxweight + 1, -1)
    print("looprange", looprange, g.number_of_nodes())
    for i in looprange:
        with Timer(time_list=encoding_times):
            encoding = generate_encoding(g, i)
            cnf = os.path.join(temp, instance + '_' + str(i) + ".cnf")
            with open(cnf, 'w') as ofile:
                ofile.write(encoding)
        sol = os.path.join(temp, instance + '_' + str(i) + ".sol")
        cmd = [cli_args.solver, '-cpu-lim={}'.format(cli_args.timeout), cnf, sol]
        # print cmd
        with Timer(time_list=solving_times):
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, err = p.communicate()
            rc = p.returncode
        sys.stderr.write('*' * 10 + '\n')
        # print output, err
        sys.stderr.write("%i %i\n" % (i - 1, rc))
        if rc == 0:
            to = True
            if lb == ub == 0:  # first timeout, record ub
                ub = i
        if rc == 20:
            if to:
                # lb = i - 2
                lb = i
            if lb == ub == 0:  # never timed out
                lb = ub = i
            return i, lb, ub, to, encoding_times, solving_times


def signal_handler(signum, frame):
    print("aborting due to signal", signum)
    print("* final treedepth ?")
    sys.exit(0)


def main(args):
    cpu_time = time.time()
    instance = args.instance
    if instance is not None:
        edge = read_edge(instance)
        g = nx.MultiGraph()
        g.add_edges_from(edge)
        instance = os.path.basename(instance)
        instance = instance.split('.')
        instance = instance[0]
    else:
        g = args.graph if "graph" in args else nx.balanced_tree(2, 2)
        # g = nx.complete_bipartite_graph(2,2)
        # g = nx.complete_graph(7)
        # g = nx.balanced_tree(2, 2)
        # g = nx.cycle_graph(15)
        # g = nx.path_graph(70)
        instance = 'random'
        # show_graph(g,6)
    args.instance = instance
    n = g.number_of_nodes()
    m = g.number_of_edges()
    buff = 0
    with Timer() as prep_timer:
        if args.preprocess:
            print("preprocessing...", file=sys.stderr)
            g = degree_one_reduction(g=g)
            g, buff = apex_vertices(g=g)
    print("* buffer verts:", buff)
    print('treedepthp2sat', instance, n, m, g.number_of_nodes(), buff)
    if args.width != -1:
        return
    ncomps = nx.number_connected_components(g)
    icomp = 1
    global_lb, global_ub = 1e9, -1
    if ncomps == 0:  # only empty graph remains after preprocessing, for loop won't be triggered
        global_lb = global_ub = 0
    for comp_nodes in nx.connected_components(g):
        subgraph = g.subgraph(comp_nodes)
        print("\ncomponent", icomp, "of", ncomps, file=sys.stderr)
        component = nx.convert_node_labels_to_integers(subgraph, first_label=0,
                                                       label_attribute="original_label")
        label_mapping = dict(component.nodes.data("original_label"))
        inverse_mapping = {v: u for u, v in label_mapping.items()}
        found_ancestries = component.graph.get("forced_ancestries", [])
        print("found ancestries:", found_ancestries)
        remapped_ancestries = []
        for v,u in found_ancestries:
            remapped_ancestries.append((inverse_mapping[v], inverse_mapping[u]))
        print("remapped ancestries:", remapped_ancestries)
        component.graph["forced_ancestries"] = remapped_ancestries
        print("weights:", component.nodes.data("weight", default=0))
        i, lb, ub, to, encoding_time, solving_time = solve_component(component, args)
        sol_file = os.path.join(args.temp, instance + '_' + str(ub + 1) + ".sol")
        decomp = decode_output(sol=sol_file, g=component, reqwidth=ub + 1,
                               return_decomp=True)
        # reapply weights
        for u, weight in component.nodes.data("weight"):
            if weight is not None: decomp.nodes[u]["weight"] = weight
        decomp = nx.relabel_nodes(decomp, label_mapping)
        print(i - 2, lb, ub, to, time.time() - cpu_time, prep_timer.duration,
              sum(encoding_time), sum(solving_time), end="")
        for j in solving_time:
            print(j, end="")
        print()
        print("* component treedepth range: [{}-{}]".format(lb, ub), file=sys.stderr)
        global_lb = min(global_lb, lb)
        global_ub = max(global_ub, ub)
        icomp += 1
    print("\n* final treedepth:", end="")
    if global_ub == global_lb:
        print(buff + global_ub, end="")
    else:
        print("[{}-{}]".format(buff + global_lb, buff + global_ub), end="")
    print("\ttotal-time: {:.2f}s".format(time.time() - cpu_time), end="")
    return buff + global_lb, buff + global_ub, decomp


# argument parser
parser = argparse.ArgumentParser(description='%(prog)s -f instance')
parser.add_argument('-f', '--file', dest='instance', action='store', type=lambda x: os.path.realpath(x),
                    default=None, help='instance')
parser.add_argument('-o', '--timeout', dest='timeout', action='store', type=int, default=900,
                    help='timeout for each SAT call')
parser.add_argument('-d', '--depth', dest='d', action='store', type=int, default=-1, help='depth')
parser.add_argument('-w', '--width', dest='width', action='store', type=int, default=-1, help='width')
parser.add_argument('-t', '--temp', dest='temp', action='store', type=str, default='./tmp',
                    help='temporary folder')
parser.add_argument('-s', '--solver', dest='solver', action='store', type=str, default='glucose',
                    help='SAT solver')  # or 'minicard_encodings_static'
parser.add_argument('-n', '--no-preprocess', dest="preprocess", action='store_false', help="Turn off preprocessing")

if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal_handler)
    main(parser.parse_args())
