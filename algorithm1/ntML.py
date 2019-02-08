import numpy as np
import matplotlib.pyplot as plt
from lpi import Expression
from lpi import Solver
from lpi.constraints import Or
from lpi.constraints import And
from termination.result import Result
from termination.result import TerminationResult
from termination.output import Output_Manager as OM


def plot(classifier, X, Y, variables):
    # if len(variables) != 2:
    #     return
    # get the separating hyperplane
    W = classifier.coef_[0]
    print(classifier.intercept_[0])
    w = list(W)
    print(w)

    if w[1] == 0:
        yy = yy_down = yy_up = np.linspace(min([Xi[1] for Xi in X]) - 2, max([Xi[1] for Xi in X]) + 2)
        a = 0
        xx = [- (classifier.intercept_[0]) / w[0]] * len(yy)
        # support vectors
        b = classifier.support_vectors_[0]
        xx_down = [b[0]] * len(yy)
        b = classifier.support_vectors_[-1]
        xx_up = [b[0]] * len(yy)
    else:
        xx = xx_down = xx_up = np.linspace(min([Xi[0] for Xi in X]) - 2, max([Xi[0] for Xi in X]) + 2)
        a = -w[0] / w[1]
        yy = a * xx - (classifier.intercept_[0]) / w[1]
        # support vectors
        b = classifier.support_vectors_[0]
        yy_down = a * xx + (b[1] - a * b[0])
        b = classifier.support_vectors_[-1]
        yy_up = a * xx + (b[1] - a * b[0])
    # plot the parallels to the separating hyperplane that pass through the

    # plot the line, the points, and the nearest vectors to the plane
    plt.plot(xx, yy, 'k-')
    plt.plot(xx_down, yy_down, 'k--')
    plt.plot(xx_up, yy_up, 'k--')
    plt.xlabel(variables[0])
    plt.ylabel(variables[1])
    plt.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1],
                s=80, facecolors='none')
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.legend()
    plt.axis('tight')
    plt.show()


def split(bad_points, good_points, variables):
    from sklearn.svm import SVC, LinearSVC
    bad_y = [False] * len(bad_points)
    good_y = [True] * len(good_points)
    X = np.array(bad_points + good_points)
    Y = np.array(bad_y + good_y)
    clf = SVC(kernel="linear")
    clf.fit(X, Y)
    OM.printif(2, clf.intercept_)
    new_phi = Expression(1 * clf.intercept_[0])
    for i in range(len(variables)):
        new_phi += Expression(1 * clf.coef_[0][i]) * Expression(variables[i])
    new_phi = new_phi >= 0
    plot(clf, X, Y, variables)
    return new_phi


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
    return ps


class MLPhi:
    def __init__(self, scc, method="disjuntive"):
        if method == "disjuntive":
            self.get = self.get_disjuntive
            self.create = self.create_disjuntive
            self.append = self.append_disjuntive
        elif method == "conjuntive":
            self.get = self.get_conjuntive
            self.create = self.create_conjuntive
            self.append = self.append_conjuntive
        else:
            raise ValueError("Unknown method for phis")
        self.phi = self.create(scc)

    def append_disjuntive(self, node, tr, value):
        self.phi[node][tr] = And([value, self.phi[node][tr]])

    def append_conjuntive(self, node, tr, value):
        self.phi[node] = And([value, self.phi[node]])

    @staticmethod
    def create_disjuntive(scc):
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
        return fancy

    @staticmethod
    def create_conjuntive(scc):
        fancy = MLPhi.create_phi_disjuntive(scc)
        f = {k: Or(fancy[k][t]) for k in fancy for t in fancy[k]}
        return f

    def get_disjuntive(self, node, tr=None, vs=None, pvs=None):
        rename = not (vs is None and pvs is None)
        if tr is None:
            if rename:
                mphi = [self.phi[node][k].renamed(vs, pvs) for k in self.phi[node]]
            else:
                mphi = [self.phi[node][k] for k in self.phi[node]]
            return Or(mphi)
        else:
            return self.phi[node][tr].renamed(vs, pvs) if rename else self.phi[node][tr]

    def get_conjuntive(self, node, tr=None, vs=None, pvs=None):
        rename = not (vs is None and pvs is None)
        return self.phi[node].renamed(vs, pvs) if rename else self.phi[node]

    def __repr__(self):
        return str(self.phi)


class ML:
    ID = "ml"
    NAME = "ml"
    DESC = "non termination using ML"

    @classmethod
    def run(cls, scc):
        Npoints = 10
        OM.printif(1, "--> with " + cls.NAME)
        global_vars = scc.get_info("global_vars")
        Nvars = int(len(global_vars) / 2)
        vs = global_vars[:Nvars]
        pvs = global_vars[Nvars:]
        phi = MLPhi(scc, method="disjuntive")
        print(phi)
        # phi = {n: v for n, v in scc.nodes.data("exit_props", default=ExpOr([]))}
        # phi = {n: ExpOr([ExpAnd(e) for e in v]) for n, v in scc.nodes.data("fancy_prop", default=[])}
        queue = [n for n in scc.get_nodes()]
        max_tries = 10
        tries = {n: max_tries for n in scc.get_nodes()}
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
                good_cons += [phi.get(node, tname)]  # phi[node][tname]
                good_cons += t["polyhedron"].get_constraints()  # ti
                good_cons += [phi.get(t["target"], vs=vs, pvs=pvs)]  # phi[ti[target]]
                bad_cons = []
                bad_cons += [phi.get(node, tname)]  # phi[node][tname]
                bad_cons += t["polyhedron"].get_constraints()  # ti
                bad_cons += [phi.get(t["target"], vs=vs, pvs=pvs).negate()]  # ¬ phi[ti[target]]
                sb = Solver()
                sb.add((And(bad_cons)))
                if sb.is_sat():
                    sg = Solver()
                    sg.add((And(good_cons)))
                    if sg.is_sat():
                        print(vs)
                        print("good")
                        print(sg)
                        good_points = generateNpoints(sg, Npoints, vs)
                        print(good_points)
                        print("bad")
                        print(sb)
                        bad_points = generateNpoints(sb, Npoints, vs)  # program terminates
                        print(bad_points)
                        new_phi = split(bad_points, good_points, vs)
                        OM.printif(2, "adding constraint for: {} (tr{})".format(node, tname), new_phi.toString(str, float))
                        phi.append(node, tname, new_phi)
                        # S = Solver()
                        # S.add((phi[node][tname]))
                        # if S.is_sat():
                        #     queue.append(node)
                        # else:
                        #     OM.printif(1, "node {} (tr {}) got unsat phi".format(node, tname))
                    else:
                        warnings = True
                        OM.printf("WARNING: bads (terminating) are sat, but goods (non-terminating) are UNSAT")
                        OM.printf("removing transition: {}".format(tname))
                        OM.printif(2, "goods", "{{{}}}".format(",\n".join([c.toString(str, str, and_symb=",\n") for c in sg.get_constraints()])))
                        scc.remove_edge(t["source"], t["target"], t["name"])
                        del phi[node][tname]
                        continue
                else:
                    OM.printif(1, "bads (terminating) are UNSAT")

        cad = ("\nphi props: {\n")
        for node in scc.get_nodes():
            cad += ("\t {} : {} \n".format(node, phi.get(node).toString(str, float, and_symb="AND\t", or_symb="\tOR")))
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
