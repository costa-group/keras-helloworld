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

    def __init__(self, classifier="linear", phi_transition=False, phi_nondeterminism=False, sampling="solver",
                 sampling_number=10, sampling_threshold=2, plot=False):
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
            self.classifier_defaultvalues["class_weight"] = {1000: 100, -1000: 1} 
        else:
            raise ValueError("Unknown method ({}) for phis".format(classifier))

        if sampling == "solver":
            self.getPoints = self.getPoints_after
            self.generatePoints = self.generatePoints_solver
        elif sampling == "lp":
            self.getPoints = self.getPoints_after
            self.generatePoints = self.generatePoints_lp
        elif sampling == "rays":
            self.getPoints = self.getPoints_before
            self.generatePoints = self.generatePoints_rays
        else:
            raise ValueError("Unknown method ({}) for sampling".format(sampling))

    def getPoints_after(self, solver, vs, lvs, all_vars, others=[]):
        bad_points = self.generatePoints(solver, vs + lvs, all_vars, tags=["_phi", "_tr", "_bad"], others=others)  # program terminates
        good_points = self.generatePoints(solver, vs + lvs, all_vars, tags=["_phi", "_tr", "_good"], others=others + bad_points)
        OM.printif(1, "goods: ", len(good_points), "bads: ", len(bad_points))
        if len(lvs) > 0:
            extra_p = self.generatePoints(solver, vs + lvs, all_vars, tags=["_phi", "_no_tr"])
            OM.printif(1, "also bad points:", len(extra_p))
            bad_points += extra_p
        return bad_points, good_points

    def getPoints_before(self, solver, vs, lvs, all_vars, others=[]):
        points = self.generatePoints(solver, vs + lvs, all_vars, tags=["_phi"])
        new_ps = []
        for p in points:
            for v in vs + lvs:
                new_p = dict(p)
                new_p[v] += 11
                if solver.is_in(new_p, names=["_phi"]):
                    new_ps.append(new_p)
                new_p = dict(p)
                new_p[v] -= 11
                if solver.is_in(new_p, names=["_phi"]):
                    new_ps.append(new_p)
        points += new_ps
        bad_points = []
        good_points = []
        for p in points:
            if p in others:
                continue
            if solver.is_in(p, names=["_phi", "_tr", "_bad"]):
                bad_points.append([p[v] for v in vs + lvs])
            elif solver.is_in(p, names=["_phi", "_tr", "_good"]):
                good_points.append([p[v] for v in vs + lvs])
            else:
                OM.printif(3, "ignoring point: ", p)
        OM.printif(1, "goods: ", len(good_points), "bads: ", len(bad_points))
        if len(lvs) > 0:
            extra_p = self.generatePoints(solver, vs + lvs, all_vars, tags=["_phi", "_no_tr"])
            OM.printif(1, "also bad points:", len(extra_p))
            bad_points += extra_p
        return bad_points, good_points

    def create_classifier(self, *args, **kwargs):
        kargs = dict(self.classifier_defaultvalues)
        kargs.update(**kwargs)
        # return SVC(kernel="linear")
        return self.classifier(*args, **kargs)

    def create_phi(self, scc):
        return MLPhi(scc, phi_method=self.phi_method)

    def generatePoints_rays(self, s, vs, all_vars, others=[], tags=None):
        from lpi import C_Polyhedron
        cons = And(s.get_constraints(tags=tags))
        from ppl import Variable
        poly = C_Polyhedron(cons, variables=vs)
        gene = poly.get_generators()
        g_points = [g for g in gene if g.is_point()]
        g_lines = [g for g in gene if g.is_line()]
        g_rays = [g for g in gene if g.is_ray()]
        rs = []
        ps = []
        for l in g_lines:
            item = {}
            item2 = {}
            for i in range(len(vs)):
                item[vs[i]] = int(l.coefficient(Variable(i)))
                item2[vs[i]] = -int(l.coefficient(Variable(i)))
            rs.append(item)
            rs.append(item2)
        for r in g_rays:
            item = {}
            for i in range(len(vs)):
                item[vs[i]] = int(r.coefficient(Variable(i)))
            rs.append(item)
        for p in g_points:
            item = {}
            for i in range(len(vs)):
                item[vs[i]] = int(p.coefficient(Variable(i)))
            ps.append((item, int(p.divisor())))

        def combine(p, rays, coeffs, vs, incr):
            point = {v: p[v] for v in vs}
            for i in range(len(coeffs)):
                for v in vs:
                    point[v] += coeffs[i] * rays[i][v]
            if coeffs[incr] < 10:
                yield from combine(p, rays, coeffs[:i] + [coeffs[i] + 1] + coeffs[i + 1:], vs, incr)
            elif incr < len(coeffs):
                yield from combine(p, rays, coeffs, vs, incr + 1)
            yield point
        points = []
        for p, d in ps:
            points.append({v: (p[v] / d) for v in vs})
            for r in rs:
                for k in range(1, 100):
                    points.append({v: ((p[v] + k * r[v]) / d) for v in vs})
        return points
        points = []
        for p, d in ps:
            points += list(combine(p, rs, [0 for __ in rs], vs, 0))
        print(points, "\n", gene)
        return points

    def generatePoints_lp(self, s, vs, all_vars, others=[], tags=None):
        from lpi import C_Polyhedron
        cons = And(s.get_constraints(tags=tags))
        cons_dnf = cons.to_DNF()
        ps = []
        queue = [[c for c in cs if c.is_linear()] for cs in cons_dnf]
        count = 0
        while len(ps) < self.sampling["N"] and len(queue) > 0:
            if count == 10:
                raise ValueError()
            cs = queue.pop()
            poly = C_Polyhedron(constraints=cs, variables=all_vars)
            if poly.is_empty():
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
            if p not in ps and p not in others:
                ps.append(p)
            else:
                count += 1
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
        try:
            new_phi = Expression(1 * clf.intercept_[0])
            for i in range(len(variables)):
                new_phi += Expression(1 * clf.coef_[0][i]) * Expression(variables[i])
            new_phi = new_phi > 0
            self.plot(clf, X, Y, variables)
            return new_phi
        except TypeError:
            return None

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
            self.remove = self.remove_disjuntive
            self.create = self.create_disjuntive
            self.append = self.append_disjuntive
        else:
            self.get = self.get_conjuntive
            self.remove = self.remove_conjuntive
            self.create = self.create_conjuntive
            self.append = self.append_conjuntive
        self.phi = self.create(scc)

    def append_disjuntive(self, node, tr, value, deterministic=True):
        if self.phi[node][tr] is None:
            return
        idx = 0 if deterministic else 1
        self.phi[node][tr][idx] = And([value, self.phi[node][tr][idx]])

    def append_conjuntive(self, node, tr, value, deterministic=True):
        idx = 0 if deterministic else 1
        self.phi[node][idx] = And([value, self.phi[node][idx]])

    def remove_disjuntive(self, node, tr):
        self.phi[node][tr] = None
        return

    def remove_conjuntive(self, node, tr):
        return

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
            if self.phi[node][k] is None:
                continue
                cs = [Expression(0) >= Expression(1)]
            else:
                if not only_deterministic:
                    cs = [self.phi[node][k][1]]
                if rename:
                    cs += [self.phi[node][k][0].renamed(vs, pvs)]
                else:
                    cs += [self.phi[node][k][0]]
            if len(cs) == 0:
                continue
            elif len(cs) == 1:
                mphi.append(cs[0])
            else:
                mphi.append(And(cs))
        if len(mphi) == 0:
            return Expression(0) >= Expression(1)
        elif len(mphi) == 1:
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
        OM.printif(1, "initial phi", phi)
        max_tries = 10
        trs = {t["name"]: t for t in scc.get_edges()}
        tr_queue = list(trs.keys())
        tr_tries = {n: max_tries for n in tr_queue}
        skiped = []
        itis = TerminationResult.UNKNOWN
        while tr_queue:
            tname = tr_queue.pop(0)
            if tr_tries[tname] == 0:
                OM.printif(1, "TOO MANY TRIES with tr: ", tname)
                skiped.append(tname)
                continue
            t = trs[tname]
            source = t["source"]
            target = t["target"]
            srcphi = phi.get(source, tname)
            s = Solver()
            s.add(srcphi, name="_phi")
            if not s.is_sat(["_phi"]):
                continue
            OM.printif(0, "Analyzing transition: ", tname)
            lvs = t["local_vars"]
            all_vars = global_vars + lvs
            s.add(t["polyhedron"].get_constraints(), name="_tr")
            if len(lvs) > 0:
                s.add(And([c for c in t["polyhedron"].get_constraints()]).negate(), name="_no_tr")
            s.add(phi.get(target, vs=vs, pvs=pvs, only_deterministic=True), name="_good")  # phi[ti[target]]
            s.add(phi.get(target, vs=vs, pvs=pvs, only_deterministic=True).negate(), name="_bad")  # not phi[ti[target]]
            goods = s.is_sat(["_phi", "_tr", "_good"])
            bads = s.is_sat(["_phi", "_tr", "_bad"]) or (len(lvs) > 0 and s.is_sat(["_phi", "_no_tr"]))
            modified = False
            mixed = False
            bad_points = []
            while goods and bads:
                if tr_tries[tname] == 0:
                    OM.printif(1, "TOO MANY TRIES with tr: ", tname)
                    skiped.append(tname)
                    mixed = True
                    break
                tr_tries[tname] -= 1
                new_bad_points, good_points = conf.getPoints(s, vs, lvs, all_vars, others=bad_points)
                bad_points += new_bad_points
                OM.printif(2, "bad:", bad_points)
                OM.printif(2, "good:", good_points)
                if len(bad_points) == 0 or len(good_points) == 0:
                    mixed = True
                    modified = True
                    OM.printif(1, "WARNING: Couldn't generate points of both groups.")
                    break
                new_phi = conf.split(bad_points, good_points, vs)
                if new_phi is None:
                    modified = True
                    mixed = True
                    OM.printif(1, "WARNING: Couldn't find a line")
                    break
                else:
                    new_phi.aproximate_coeffs(max_coeff=1e10, max_dec=10)
                    OM.printif(1, "split found:", new_phi.toString(str,str))
                    s.add(new_phi, name="_phi")
                    modified = True
                    phi.append(source, tname, new_phi)
                goods = s.is_sat(["_phi", "_tr", "_good"])
                bads = s.is_sat(["_phi", "_tr", "_bad"])
            if modified:
                for k in trs:
                    if trs[k]["target"] == source and trs[k]["name"] not in tr_queue:
                        if trs[k]["name"] == tname and mixed:
                            continue
                        tr_queue.append(trs[k]["name"])
            if mixed:
                OM.printf("removing transition: {}".format(tname))
                scc.remove_edge(t["source"], t["target"], t["name"])
                phi.remove(source, tname)
            elif not bads:
                OM.printif(1, "{}: bads (terminating) are UNSAT".format(tname))
            elif not goods:
                OM.printf("{}: bads (terminating) are SAT, but goods (non-terminating) are UNSAT".format(tname))
                OM.printf("removing transition: {}".format(tname))
                scc.remove_edge(t["source"], t["target"], t["name"])
                phi.remove(source, tname)

        nonterminate = False
        for sub_scc in scc.get_scc():
            if len(sub_scc.get_edges()) == 0:
                continue
            sub_nont = True
            for t in sub_scc.get_edges():
                source = t["source"]
                target = t["target"]
                tname = t["name"]
                OM.printif(1, "Checking transition: ", tname)
                srcphi = phi.get(source, tname)
                s = Solver()
                s.add(srcphi, name="_phi")
                lvs = t["local_vars"]
                all_vars = global_vars + t["local_vars"]
                s.add(t["polyhedron"].get_constraints(), name="_tr")
                s.add(phi.get(target, vs=vs, pvs=pvs, only_deterministic=True), name="_good")  # phi[ti[target]]
                s.add(phi.get(target, vs=vs, pvs=pvs, only_deterministic=True).negate(), name="_bad")  # not phi[ti[target]]
                goods = s.is_sat(["_phi", "_tr", "_good"])
                bads = s.is_sat(["_phi", "_tr", "_bad"])
                OM.printif(1, "goods: {}, bads: {}".format(goods, bads))
                if not goods or bads:
                    sub_nont = False
                    break
            if sub_nont:
                nonterminate = True
                for node in sub_scc.get_nodes():
                    OM.printif(2, "\t {} :".format(node))
                    for t in sub_scc.get_edges(source=node):
                        tname = t["name"]
                        OM.printif(2, "\t\t {} : {}".format(tname, phi.get(node, tname).toString(str, str, and_symb="AND", or_symb="OR")))
                        OM.printif(3, "\t\t {}(rounded) : {}".format(tname, phi.get(node, tname).toString(str, mround, and_symb="AND", or_symb="OR")))
                break
        if len(skiped) > 0:
            OM.printf("some transitions ({}) where ignored after {} iterations.".format(skiped, max_tries))

        if nonterminate:
            itis = TerminationResult.NONTERMINATE
            OM.printf("phi props: {")
            for node in scc.get_nodes():
                OM.printf("\t {} :".format(node))
                for edge in scc.get_edges(source=node):
                    tname = edge["name"]
                    OM.printf("\t\t {} : {}".format(tname, phi.get(node, tname).toString(str, str, and_symb="AND", or_symb="OR")))
                    OM.printif(2, "\t\t {}(rounded) : {}".format(tname, phi.get(node, tname).toString(str, mround, and_symb="AND", or_symb="OR")))
            OM.printf("}")
        OM.printf("Answer: {}".format(itis))
        max_iterations_needed = max_tries - min([tr_tries[k] for k in tr_tries])
        OM.printf("Iterations: {}".format(max_iterations_needed))
        OM.printif(1, "Graph:")
        for e in scc.get_edges():
            OM.printif(1, e["name"], ":", e["source"], "-->", e["target"])

        response = Result()
        response.set_response(status=itis)
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
