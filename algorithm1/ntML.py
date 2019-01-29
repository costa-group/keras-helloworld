
from z3 import sat
from z3 import Real
from z3 import Solver, And, Or
from main import printif
from termination.algorithm.utils import generate_prime_names
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
from lpi import Expression
from lpi import C_Polyhedron
from lpi.constraints import Or as ExpOr
from lpi.constraints import And as ExpAnd


def toz3(c):
    if isinstance(c, list):
        if len(c) > 0 and isinstance(c[0], list):
            ors = []
            for a in c:
                ands = [toz3(e) for e in a]
                if len(ands) == 1:
                    ors.append(ands[0])
                elif len(ands) > 1:
                    ors.append(And(ands))
            if len(ors) == 1:
                return ors[0]
            elif len(ors) > 1:
                return Or(ors)
            else:
                return None
        else:
            if len(c) == 1:
                return toz3(c[0])
            return And([toz3(e) for e in c])
    return c.get(Real, float, ignore_zero=True, toOr=Or, toAnd=And)


def generateNpoints(s, N, vs):
    s.push()
    ps = []
    rvs = [Real(v) for v in vs]
    while s.check() == sat:
        m = s.model()
        p = [float(int(str(m[v].numerator())) / int(str(m[v].denominator()))) for v in rvs]
        ps.append(p)
        if len(ps) == N:
            break
        exp = [v != m[v] for v in rvs]
        if len(exp) > 0:
            s.add(Or(exp))
        else:
            raise Exception("something went wrong...")
    s.pop()
    return ps


class ML:
    ID = "ml"
    NAME = "ml"
    DESC = "non termination using ML"

    def __init__(self, properties={}):
        self.props = properties

    def run(self, cfg):
        Npoints = 10
        printif(1, "--> with " + self.NAME)
        global_vars = cfg.get_info("global_vars")
        Nvars = int(len(global_vars) / 2)
        vs = global_vars[:Nvars]
        pvs = global_vars[Nvars:]
        phi = {n: ExpOr([ExpAnd(e) for e in v]) for n, v in cfg.nodes.data("fancy_prop", default=[])}
        queue = [n for n in phi if phi[n].len() > 0]
        bad_points = []
        bad_y = []
        while len(queue) > 0:
            node = queue.pop()
            good_cons = [toz3(phi[node])]  # phi[node] ^ ti ^ phi[ti[target]]  ^ ...
            bad_cons = []  # phi[node] ^ ti ^ ¬ phi[ti[target]] V ...
            taken_vars = []
            for t in cfg.get_edges(source=node):
                print(t["name"])
                lvs = t["local_vars"]
                taken_vars += lvs
                newprimevars = generate_prime_names(vs, taken_vars)
                taken_vars += newprimevars
                good_cons.append(toz3(t["polyhedron"].get_constraints(vs + newprimevars + lvs)))
                good_cons.append(toz3(phi[t["target"]].renamed(vs, newprimevars)))
                bad_cons.append(And(toz3(phi[node]),
                                    toz3(t["polyhedron"].get_constraints()),
                                    toz3(phi[t["target"]].negate().renamed(vs, pvs))))
                print("good cons:", good_cons)
                print("bad cons:", bad_cons)
            Sb = Solver()
            Sb.add(Or(bad_cons))
            if Sb.check() == sat:
                # bad_points = [[0, i * -100] for i in range(3, 3 + Npoints)]
                bad_points = generateNpoints(Sb, Npoints, vs)  # program terminates
                bad_y = [False] * Npoints
                Sg = Solver()
                Sg.add(good_cons)
                if Sg.check() == sat:
                    # good_points = [[0, i * 100] for i in range(3, 3 + Npoints)]
                    good_points = generateNpoints(Sg, Npoints, vs)
                    good_y = [True] * Npoints
                    X = np.array(bad_points + good_points)
                    Y = np.array(bad_y + good_y)
                    clf = SVC(kernel="linear")
                    clf.fit(X, Y)
                    new_phi = Expression(1 * clf.intercept_[0])
                    for i in range(Nvars):
                        new_phi += Expression(1 * clf.coef_[0][i]) * Expression(vs[i])
                    new_phi = new_phi >= 0
                    print("adding constraint for: {}".format(node), new_phi.toString(str, float))
                    if Nvars == 2:
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
                        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
                        plt.legend()
                        plt.axis('tight')
                        plt.show()
                    phi[node] = ExpAnd(new_phi, phi[node])
                    S = Solver()
                    S.add(toz3(phi[node]))
                    if S.check() == sat:
                        queue.append(node)
                    else:
                        print("node {}, got unsat phi".format(node))
                else:
                    print("WARNING: bads (terminating) are sat, but goods (non-terminating) are UNSAT")
            else:
                print("bads (terminating) are UNSAT")
        print("phi props: {")
        for node in phi:
            print("\t", node, ":", phi[node].toString(str, float, and_symb=",\n\t\t", or_symb="\n\t\tOR"))
        print("}")
        return {"status": "?"}
