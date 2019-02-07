import numpy as np
from sklearn.svm import SVC, LinearSVC
# from .manager import Algorithm
import matplotlib.pyplot as plt
from lpi import Expression
from lpi import Solver
from lpi.constraints import Or
from lpi.constraints import And
from termination.result import Result
from termination.result import TerminationResult
from termination.output import Output_Manager as OM
from termination.algorithm.utils import generate_prime_names


class ML_:
    ID = "ml"
    NAME = "ml"
    DESC = "non termination using ML"

    @staticmethod
    def parse_props(properties, token):
        props = dict(properties)
        ops = list(token)
        props["phi"] = "one"
        props["roundup"] = False
        for o in ops:
            if o == "one" or o == "dis":
                props["phi"] = o
            elif o == "up":
                props["roundup"] = True
        return props

    @staticmethod
    def run(cls, cfg, properties={}, token=[]):
        pass
        # props = .parse_props(properties, token)


def set_fancy_props_neg(cfg, scc):
    fancy = {}
    nodes = scc.get_nodes()
    gvs = cfg.get_info("global_vars")
    Nvars = int(len(gvs) / 2)
    vs = gvs[:Nvars]
    for node in nodes:
        ors = []
        for tr in cfg.get_edges(source=node):
            if tr["target"] in nodes:
                continue
            cs = tr["polyhedron"].project(vs).get_constraints()
            if len(cs) == 0:
                continue
            if len(cs) == 1:
                ors.append(cs[0])
            else:
                ors.append(And(cs))
        if len(ors) == 0:
            fancy[node] = Or(Expression(0) == Expression(0))
        else:
            fancy[node] = Or([o.negate() for o in ors])
    OM.printf("exit conditions: \n", fancy, "\n ===========================")
    scc.set_nodes_info(fancy, "exit_props")


def set_fancy_props(scc):
    fancy = {}
    nodes = scc.get_nodes()
    gvs = scc.get_info("global_vars")
    Nvars = int(len(gvs) / 2)
    vs = gvs[:Nvars]
    for node in nodes:
        ors = {}
        for tr in scc.get_edges(source=node):
            cs = tr["polyhedron"].project(vs).get_constraints()
            tname = tr["name"]
            if len(cs) == 0:
                ors[tname] = Expression(0) == Expression(0)
            if len(cs) == 1:
                ors[tname] = cs[0]
            else:
                ors[tname] = And(cs)
        fancy[node] = ors
    OM.printf("exit conditions: \n", fancy, "\n ===========================")
    # scc.set_nodes_info(fancy, "exit_props")
    return fancy


def generateNpoints(s, N, vs):
    s.push()
    ps = []
    # rvs = [Real(v) for v in vs]
    while s.is_sat():
        m, __ = s.get_point(vs)
        p = []
        exp = []
        for v in vs:
            if m[v] is None:
                pi = 0.0
            else:
                exp.append(Expression(v) > Expression(m[v]))
                exp.append(Expression(v) < Expression(m[v]))
                pi = m[v]  # float(int(str(m[v].numerator())) / int(str(m[v].denominator())))
            p.append(pi)
        ps.append(p)
        if len(ps) == N:
            break
        if len(exp) > 0:
            s.add(Or(exp))
        else:
            raise Exception("something went wrong...")
    s.pop()
    print(ps)
    return ps


class ML:
    ID = "ml"
    NAME = "ml"
    DESC = "non termination using ML"

    def __init__(self, properties={}):
        self.props = properties

    @classmethod
    def run(cls, scc):
        Npoints = 10
        OM.printif(1, "--> with " + cls.NAME)
        global_vars = scc.get_info("global_vars")
        Nvars = int(len(global_vars) / 2)
        vs = global_vars[:Nvars]
        pvs = global_vars[Nvars:]
        fancy = set_fancy_props(scc)
        phi = {n: fancy[n] for n in fancy}
        # phi = {n: v for n, v in scc.nodes.data("exit_props", default=ExpOr([]))}
        # phi = {n: ExpOr([ExpAnd(e) for e in v]) for n, v in scc.nodes.data("fancy_prop", default=[])}
        queue = [n for n in phi if len(phi[n].keys()) > 0]
        bad_points = []
        bad_y = []
        max_tries = 10
        tries = {n: max_tries for n in phi}
        skiped = []
        warnings = False
        itis = TerminationResult.NONTERMINATE
        while len(queue) > 0:
            node = queue.pop()
            if tries[node] == 0:
                OM.printf("TOO MANY TRIES with node: ", node)
                skiped.append(node)
                continue
            tries[node] -= 1

            for t in scc.get_edges(source=node):
                tname = t["name"]
                print(tname)
                good_cons = []
                good_cons += [phi[node][tname]]  # phi[node][tname]
                good_cons += t["polyhedron"].get_constraints()  # ti
                good_cons += [Or([phi[t["target"]][k].renamed(vs, pvs) for k in phi[t["target"]]])]  # phi[ti[target]]
                bad_cons = []
                bad_cons += [phi[node][tname]]  # phi[node][tname]
                bad_cons += t["polyhedron"].get_constraints()  # ti
                bad_cons += [And([phi[t["target"]][k].negate().renamed(vs, pvs) for k in phi[t["target"]]])]  # ¬ phi[ti[target]]
                OM.printif(2, "good cons:", good_cons)
                OM.printif(2, "bad cons:", bad_cons)
                sb = Solver()
                sb.add((And(bad_cons)))
                OM.printif(2, "bad", sb)
                if sb.is_sat():
                    sg = Solver()
                    sg.add((And(good_cons)))
                    if sg.is_sat():
                        bad_points = generateNpoints(sb, Npoints, vs)  # program terminates
                        bad_y = [False] * len(bad_points)
                        good_points = generateNpoints(sg, Npoints, vs)
                        good_y = [True] * len(good_points)
                        X = np.array(bad_points + good_points)
                        Y = np.array(bad_y + good_y)
                        clf = SVC(kernel="linear")
                        clf.fit(X, Y)
                        OM.printif(2, clf.intercept_)
                        new_phi = Expression(1 * clf.intercept_[0])
                        for i in range(Nvars):
                            new_phi += Expression(1 * clf.coef_[0][i]) * Expression(vs[i])
                        new_phi = new_phi >= 0
                        OM.printif(2, "adding constraint for: {} (tr{})".format(node, tname), new_phi.toString(str, float))
                        if False and Nvars == 2:
                            # get the separating hyperplane
                            w = clf.coef_[0]
                            a = -w[0] / w[1]
                            xx = np.linspace(2, 80)
                            yy = a * xx - (clf.intercept_[0]) / w[1]
                            # plot the parallels to the separating hyperplane that pass through the
                            # support vectors
                            b = clf.support_vectors_[0]
                            yy_down = a * xx + (b[1] - a * b[0])
                            b = clf.support_vectors_[-1]
                            yy_up = a * xx + (b[1] - a * b[0])
                            # plot the line, the points, and the nearest vectors to the plane
                            plt.plot(xx, yy, 'k-')
                            plt.plot(xx, yy_down, 'k--')
                            plt.plot(xx, yy_up, 'k--')
                            plt.xlabel(vs[0])
                            plt.ylabel(vs[1])
                            plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                                        s=80, facecolors='none')
                            plt.scatter(X[:, 0], X[:, 1], c=Y)
                            plt.legend()
                            plt.axis('tight')
                            plt.show()
                        phi[node][tname] = And([new_phi, phi[node][tname]])
                        S = Solver()
                        S.add((phi[node][tname]))
                        if S.is_sat():
                            queue.append(node)
                        else:
                            OM.printif(1, "node {} (tr {}) got unsat phi".format(node, tname))
                    else:
                        warnings = True
                        OM.printf("WARNING: bads (terminating) are sat, but goods (non-terminating) are UNSAT")
                        OM.printf("removing transition: {}".format(tname))
                        OM.printif(2, "goods", "{{{}}}".format(",\n".join([c.toString(str, str, and_symb=",\n") for c in sg.get_constraints()])))
                        scc.remove_edge(t["source"], t["target"], t["name"])
                        del phi[node][tname]
                        continue
                        # scc.remove_edge()
                        itis = TerminationResult.UNKNOWN
                else:
                    OM.printif(1, "bads (terminating) are UNSAT")

        cad = ("\nphi props: {\n")
        for node in phi:
            cad += ("\t {} : {} \n".format(node, Or([phi[node][tname]
                                                     for tname in phi[node]]).toString(str, float, and_symb="AND\t", or_symb="\tOR")))
        cad += ("\n}\n")
        if len(skiped) > 0:
            itis = TerminationResult.UNKNOWN
            cad += ("some nodes ({}) where ignored after {} iterations. \n".format(skiped, max_tries))
        if warnings:
            cad += ("Look at log.. there were WARNINGS\n")
        OM.printf(cad)
        response = Result()
        response.set_response(status=itis, info=cad)
        return response

    @classmethod
    def generate(cls, data):
        if len(data) != 1:
            return None
        if data[0] == cls.ID:
            try:
                c = cls()
            except Exception as e:
                raise Exception() from e
            return c
        return None

    @classmethod
    def use_close_walk(cls):
        return False

    @classmethod
    def description(cls, long=False):
        desc = str(cls.ID)
        if long:
            desc += ": " + str(cls.DESC)
        return desc

    def __repr__(self):
        return self.NAME

    def set_prop(self, key, value):
        self.props[key] = value

    def get_prop(self, key):
        return self.props[key]

    def has_prop(self, key):
        return key in self.props

    def get_name(self):
        return self.NAME


class ML2(ML):
    ID = "ml2"
    NAME = "ml2"
    DESC = "non termination using ML"

    def __init__(self, properties={}):
        self.props = properties

    def run(self, scc):
        Npoints = 10
        OM.printif(1, "--> with " + self.NAME)
        global_vars = scc.get_info("global_vars")
        Nvars = int(len(global_vars) / 2)
        vs = global_vars[:Nvars]
        pvs = global_vars[Nvars:]
        fancy = set_fancy_props(scc)
        phi = {n: ExpOr([fancy[n][k] for k in fancy[n]]) for n in fancy}
        # phi = {n: v for n, v in scc.nodes.data("exit_props", default=ExpOr([]))}
        # phi = {n: ExpOr([ExpAnd(e) for e in v]) for n, v in scc.nodes.data("fancy_prop", default=[])}
        queue = [n for n in phi if phi[n].len() > 0]
        bad_points = []
        bad_y = []
        max_tries = 10
        tries = {n: max_tries for n in phi}
        skiped = []
        warnings = False
        itis = TerminationResult.NONTERMINATE
        while len(queue) > 0:
            node = queue.pop()
            if tries[node] == 0:
                OM.printf("TOO MANY TRIES with node: ", node)
                skiped.append(node)
                continue
            tries[node] -= 1
            good_cons = [toz3(phi[node])]  # phi[node] ^ ti ^ phi[ti[target]]  ^ ...
            bad_cons = []  # phi[node] ^ ti ^ ¬ phi[ti[target]] V ...
            taken_vars = []
            for t in scc.get_edges(source=node):
                OM.printif(1, t["name"])
                lvs = t["local_vars"]
                taken_vars += lvs
                newprimevars = generate_prime_names(vs, taken_vars)
                taken_vars += newprimevars
                good_cons.append(toz3(t["polyhedron"].get_constraints(vs + newprimevars + lvs)))
                good_cons.append(toz3(phi[t["target"]].renamed(vs, newprimevars)))
                bad_cons.append(And(toz3(phi[node]),
                                    toz3(t["polyhedron"].get_constraints()),
                                    toz3(phi[t["target"]].negate().renamed(vs, pvs))))
                OM.printif(2, "good cons:", good_cons)
                OM.printif(2, "bad cons:", bad_cons)
            Sb = Solver()

            Sb.add(simplify(Or(bad_cons)))
            OM.printif(2, "bad", Sb)
            if Sb.check() == sat:
                # bad_points = [[0, i * -100] for i in range(3, 3 + Npoints)]
                bad_points = generateNpoints(Sb, Npoints, vs)  # program terminates
                bad_y = [False] * Npoints
                Sg = Solver()
                Sg.add(And(good_cons))
                if Sg.check() == sat:
                    # good_points = [[0, i * 100] for i in range(3, 3 + Npoints)]
                    good_points = generateNpoints(Sg, Npoints, vs)
                    good_y = [True] * Npoints
                    X = np.array(bad_points + good_points)
                    Y = np.array(bad_y + good_y)
                    clf = SVC(kernel="linear")
                    clf.fit(X, Y)
                    OM.printif(2, clf.intercept_)
                    new_phi = Expression(1 * clf.intercept_[0])
                    for i in range(Nvars):
                        new_phi += Expression(1 * clf.coef_[0][i]) * Expression(vs[i])
                    new_phi = new_phi >= 0
                    OM.printif(2, "adding constraint for: {}".format(node), new_phi.toString(str, float))
                    if False and Nvars == 2:
                        # get the separating hyperplane
                        w = clf.coef_[0]
                        a = -w[0] / w[1]
                        xx = np.linspace(2, 80)
                        yy = a * xx - (clf.intercept_[0]) / w[1]
                        # plot the parallels to the separating hyperplane that pass through the
                        # support vectors
                        b = clf.support_vectors_[0]
                        yy_down = a * xx + (b[1] - a * b[0])
                        b = clf.support_vectors_[-1]
                        yy_up = a * xx + (b[1] - a * b[0])
                        # plot the line, the points, and the nearest vectors to the plane
                        plt.plot(xx, yy, 'k-')
                        plt.plot(xx, yy_down, 'k--')
                        plt.plot(xx, yy_up, 'k--')
                        plt.xlabel(vs[0])
                        plt.ylabel(vs[1])
                        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                                    s=80, facecolors='none')
                        plt.scatter(X[:, 0], X[:, 1], c=Y)
                        plt.legend()
                        plt.axis('tight')
                        plt.show()
                    phi[node] = ExpAnd(new_phi, phi[node])
                    S = Solver()
                    S.add(toz3(phi[node]))
                    if S.check() == sat:
                        queue.append(node)
                    else:
                        OM.printif(1, "node {}, got unsat phi".format(node))
                else:
                    warnings = True
                    OM.printf("WARNING: bads (terminating) are sat, but goods (non-terminating) are UNSAT")
                    itis = TerminationResult.UNKNOWN
            else:
                OM.printif(1, "bads (terminating) are UNSAT")
        cad = ("\nphi props: {\n")
        for node in phi:
            cad += ("\t {} : {} \n".format(node, phi[node].toString(str, float, and_symb="AND\t", or_symb="\tOR")))
        cad += ("\n}\n")
        if len(skiped) > 0:
            itis = TerminationResult.UNKNOWN
            cad += ("some nodes ({}) where ignored after {} iterations. \n".format(skiped, max_tries))
        if warnings:
            cad += ("Look at log.. there were WARNINGS\n")
        OM.printf(cad)
        response = Result()
        response.set_response(status=itis, info=cad)
        return response
