import numpy as np
import matplotlib.pyplot as plt
from lpi import Expression
from lpi import Solver
from lpi.constraints import Or
from lpi.constraints import And
from termination.result import Result
from termination.result import TerminationResult
from termination.output import Output_Manager as OM


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


class MLConf:

    def __init__(self, classifier="linear", phi_transition=False, phi_nondeterminism=False, sampling="solver", sampling_number=10, sampling_threshold=2, plot=False):
        from sklearn.svm import SVC, LinearSVC
        self.classifier_plot = plot
        self.phi_method = (phi_transition, phi_nondeterminism)
        self.sampling = {"threshold": sampling_threshold, "N": sampling_number}
        self.classifier_defaultvalues = {}
        if classifier == "linear":
            self.classifier = LinearSVC
        elif classifier == "perceptron":
            self.classifier = Perceptron
        elif classifier == "svc":
            self.classifier = SVC
            self.classifier_defaultvalues["kernel"] = "linear"
        else:
            raise ValueError("Unknown method ({}) for phis".format(classifier))

        if sampling == "solver":
            self.generatePoints = self.generatePoints_solver
        elif sampling == "lp":
            self.generatePoints = self.generatePoints_lp
        else:
            raise ValueError("Unknown method ({}) for sampling".format(sampling))

    def create_classifier(self, *args, **kwargs):
        kargs = dict(self.classifier_defaultvalues)
        kargs.update(**kwargs)
        return self.classifier(*args, **kargs)

    def create_phi(self, scc):
        return MLPhi(scc, phi_method=self.phi_method)

    def generatePoints_lp(self, s, vs, all_vars, other=[], tags=None):
        from lpi import C_Polyhedron
        cons = And(s.get_constraints())
        print(cons)
        cons_dnf = cons.to_DNF()
        ps = []
        queue = [[c for c in cs if c.is_linear()] for cs in cons_dnf]
        while len(ps) < self.sampling["N"] and len(queue) > 0:
            print("iteration on generate")
            cs = queue.pop()
            poly = C_Polyhedron(constraints=cs, variables=all_vars)
            if poly.is_empty():
                print("ppl empty", cs)
                continue
            m, __ = poly.get_point()
            p = []
            exp = []
            for v in vs:
                if v not in m or m[v] is None:
                    pi = 0.0
                else:
                    pi = m[v]
                    exp.append(Expression(v) > Expression(pi) + self.sampling["threshold"])
                    exp.append(Expression(v) < Expression(pi) - self.sampling["threshold"])
                p.append(pi)
            if p not in ps:
                ps.append(p)
            if len(ps) == self.sampling["N"]:
                break
            for e in exp:
                queue.append(list(cs) + [e])
        return ps

    def generatePoints_solver(self, s, vs, all_vars=[], others=[], tags=None):
        s.push()
        ps = []
        # rvs = [Real(v) for v in vs]
        while len(ps) < self.sampling["N"] and s.is_sat(tags):
            m, __ = s.get_point(vs, tags)
            print("->", m)
            p = []
            exp = []
            for v in vs:
                if v not in m:
                    print(v)
                if m[v] is None:
                    pi = 0.0
                else:
                    pi = m[v]
                    exp.append(Expression(v) - self.sampling["threshold"] > Expression(pi))
                    exp.append(Expression(v) + self.sampling["threshold"] < Expression(pi))
                p.append(pi)
            if p not in others:
                print("ignore", p)
                ps.append(p)
            if len(ps) == self.sampling["N"]:
                break
            final_exp = []
            for e in exp:
                e.aproximate_coeffs(max_coeff=1e10, max_dec=10)
                if e.is_true() or e.is_false():
                    continue
                final_exp.append(e)
            if len(final_exp) > 0:
                s.add(Or(final_exp))
            else:
                break
        s.pop()
        return ps

    def split(self, bad_points, good_points, variables):
        bad_y = [-1000] * len(bad_points)
        good_y = [1000] * len(good_points)
        X = np.array(bad_points + good_points)
        Y = np.array(bad_y + good_y)
        clf = self.create_classifier()
        clf.fit(X, Y)
        OM.printif(2, clf.intercept_)
        new_phi = Expression(1 * clf.intercept_[0])
        for i in range(len(variables)):
            new_phi += Expression(1 * clf.coef_[0][i]) * Expression(variables[i])
        new_phi = new_phi >= 0
        self.plot(clf, X, Y, variables)
        return new_phi

    def plot(self, classifier, X, Y, variables):
        if not self.classifier_plot or len(variables) != 2:
            return
        w = classifier.coef_[0]
        if w[1] == 0:
            yy = np.linspace(min([Xi[1] for Xi in X]) - 2, max([Xi[1] for Xi in X]) + 2)
            # yy_down = yy_up = yy
            a = 0
            xx = [- (classifier.intercept_[0]) / w[0]] * len(yy)
            # support vectors
            # b = classifier.support_vectors_[0]
            # xx_down = [b[0]] * len(yy)
            # b = classifier.support_vectors_[-1]
            # xx_up = [b[0]] * len(yy)
        else:
            xx = np.linspace(min([Xi[0] for Xi in X]) - 2, max([Xi[0] for Xi in X]) + 2)
            # xx_down = xx_up = xx
            a = -w[0] / w[1]
            yy = a * xx - (classifier.intercept_[0]) / w[1]
            # support vectors
            # b = classifier.support_vectors_[0]
            # yy_down = a * xx + (b[1] - a * b[0])
            # b = classifier.support_vectors_[-1]
            # yy_up = a * xx + (b[1] - a * b[0])
        plt.plot(xx, yy, 'k-')
        # plt.plot(xx_down, yy_down, 'k--')
        # plt.plot(xx_up, yy_up, 'k--')
        plt.xlabel(variables[0])
        plt.ylabel(variables[1])
        # plt.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1],
        #            s=80, facecolors='none')
        plt.scatter(X[:, 0], X[:, 1], c=Y)
        plt.legend()
        plt.axis('tight')
        plt.show()


class MLPhi:
    def __init__(self, scc, phi_method=(False, False)):
        if phi_method[0]:
            self.get = self.get_disjuntive
            self.create = self.create_disjuntive
            self.append = self.append_disjuntive
        else:
            self.get = self.get_conjuntive
            self.create = self.create_conjuntive
            self.append = self.append_conjuntive
        self.phi = self.create(scc)

    def append_disjuntive(self, node, tr, value, deterministic=True):
        idx = 0 if deterministic else 1
        self.phi[node][tr][idx] = And([value, self.phi[node][tr][idx]])

    def append_conjuntive(self, node, tr, value, deterministic=True):
        idx = 0 if deterministic else 1
        self.phi[node][idx] = And([value, self.phi[node][idx]])

    @classmethod
    def create_disjuntive(cls, scc):
        fancy = {}
        nodes = scc.get_nodes()
        gvs = scc.get_info("global_vars")
        Nvars = int(len(gvs) / 2)
        vs = gvs[:Nvars]
        for node in nodes:
            ors = {}
            for tr in scc.get_edges(source=node):
                tname = tr["name"]
                lvs = tr["local_vars"]
                cs = tr["polyhedron"].project(vs).get_constraints()
                cs_nd = []
                if len(lvs) > 0:
                    cs_nd = tr["polyhedron"].project(lvs).get_constraints()
                if len(cs) == 0:
                    cs = Expression(0) == Expression(0)
                elif len(cs) == 1:
                    cs = cs[0]
                else:
                    cs = And(cs)
                if len(cs_nd) == 0:
                    cs_nd = Expression(0) == Expression(0)
                elif len(cs_nd) == 1:
                    cs_nd = cs_nd[0]
                else:
                    cs_nd = And(cs_nd)
                ors[tname] = [cs, cs_nd]
            fancy[node] = ors
        return fancy

    @classmethod
    def create_conjuntive(cls, scc):
        fancy = cls.create_disjuntive(scc)
        f = {k: (Or(fancy[k][t][0]), Or(fancy[k][t][1])) for k in fancy for t in fancy[k]}
        return f

    def get_disjuntive(self, node, tr=None, only_deterministic=False, vs=None, pvs=None):
        rename = not (vs is None and pvs is None)
        if tr is None:
            trs = list(self.phi[node].keys())
        else:
            trs = [tr]
        mphi = []
        for k in trs:
            cs = []
            if not only_deterministic:
                cs = [self.phi[node][k][1]]
            if rename:
                cs += [self.phi[node][k][0].renamed(vs, pvs)]
            else:
                cs += [self.phi[node][k][0]]
            if len(cs) == 1:
                mphi.append(cs[0])
            else:
                mphi.append(And(cs))
        if len(mphi) == 1:
            return mphi[0]
        else:
            return Or(mphi)

    def get_conjuntive(self, node, tr=None, only_deterministic=False, vs=None, pvs=None):
        rename = not (vs is None and pvs is None)
        if only_deterministic:
            return self.phi[node][0].renamed(vs, pvs) if rename else self.phi[node][0]
        else:
            return And(self.phi[node][0].renamed(vs, pvs), self.phi[node][1]) if rename else And(self.phi[node][0], self.phi[node][1])

    def __repr__(self):
        return str(self.phi)


class ML:
    ID = "ml"
    NAME = "ml"
    DESC = "non termination using ML"

    @classmethod
    def run(cls, scc, config):
        OM.printif(1, "--> with " + cls.NAME)
        mround = lambda x: round(x, 4)
        global_vars = scc.get_info("global_vars")
        Nvars = int(len(global_vars) / 2)
        vs = global_vars[:Nvars]
        pvs = global_vars[Nvars:]
        conf = MLConf(**config)
        phi = conf.create_phi(scc)
        OM.printf("initial phi", phi)
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
                lvs = t["local_vars"]
                all_vars = global_vars + t["local_vars"]
                OM.printif(1, "Analyzing transition: ", tname)
                s = Solver()
                s.add(phi.get(node, tname))
                s.add(t["polyhedron"].get_constraints())
                s.add(phi.get(t["target"], vs=vs, pvs=pvs, only_deterministic=True), name="_good")
                s.add(phi.get(t["target"], vs=vs, pvs=pvs, only_deterministic=True).negate(), name="_bad")
                # good_cons = []
                # good_cons += [phi.get(node, tname)]  # phi[node][tname]
                # good_cons +=   # ti
                # good_cons += []  # phi[ti[target]]
                # bad_cons = []
                # bad_cons += [phi.get(node, tname)]  # phi[node][tname]
                # bad_cons += t["polyhedron"].get_constraints()  # ti
                # bad_cons += [phi.get(t["target"], vs=vs, pvs=pvs, only_deterministic=True).negate()]  # ¬ phi[ti[target]]
                # sb = Solver()
                # sb.add((And(bad_cons)))
                if s.is_sat(["good"]):
                    if s.is_sat(["bad"]):
                        bad_points = conf.generatePoints(s, vs+lvs, all_vars, tags=["_bad"])  # program terminates
                        good_points = conf.generatePoints(s, vs+lvs, all_vars, tags=["_good"], others=bad_points)
                        print("bads", bad_points, "goods", good_points)
                        for p in good_points:
                            if p in bad_points:
                                OM.printf("FIX NON-DETERMINISM MANUALLY", tname, "point: ", {v: pi for v, pi in zip(vs, p)})
                        new_phi = conf.split(bad_points, good_points, vs)
                        OM.printif(2, "original constraint".format(node, tname), new_phi)
                        OM.printif(2, "rounded constraint".format(node, tname), new_phi.toString(str, mround))
                        new_phi.aproximate_coeffs(max_coeff=1e10, max_dec=10)
                        OM.printif(2, "constraint added to: {} (tr {})".format(node, tname), new_phi.toString(str, str))
                        old = phi.get(node, tname, only_deterministic=True)
                        phi.append(node, tname, new_phi)
                        S = Solver()
                        S.add((phi.get(node, tname, only_deterministic=True).negate()))
                        S.add(old)
                        if S.is_sat():
                            queue.append(node)
                        else:
                            OM.printif(1, "node {} (tr {}) got unsat phi".format(node, tname))
                            itis = TerminationResult.UNKNOWN
                    else:
                        warnings = True
                        OM.printf("WARNING: bads (terminating) are sat, but goods (non-terminating) are UNSAT")
                        OM.printf("removing transition: {}".format(tname))
                        OM.printif(2, "goods", "{{{}}}".format(",\n".join([c.toString(str, str, and_symb=",\n") for c in sg.get_constraints()])))
                        scc.remove_edge(t["source"], t["target"], t["name"])
                        # phi.remove(node, tname)
                        continue
                else:
                    OM.printif(1, "bads (terminating) are UNSAT")

        cad = ("\nphi props: {\n")
        for node in scc.get_nodes():
            cad += ("\t {} : {} \n".format(node, phi.get(node).toString(str, float, and_symb="AND", or_symb="OR")))
            cad += ("\t {} : {} \n".format(node, phi.get(node).toString(str, mround, and_symb="AND", or_symb="OR")))
        cad += ("\n}\n")
        if len(skiped) > 0:
            itis = TerminationResult.UNKNOWN
            cad += ("some nodes ({}) where ignored after {} iterations. \n".format(skiped, max_tries))
        if warnings:
            cad += ("Look at log.. there were WARNINGS\n")
        cad += ("Answer: {}\n".format(itis))
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


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.

    """
    def __init__(self, eta=0.2, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for __ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        self.coef_ = [list(self.w_[1:])]
        self.intercept_ = [-self.w_[0]]
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
