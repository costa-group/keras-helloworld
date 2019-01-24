
from z3 import sat
from z3 import Real
from z3 import Solver, And, Or
from main import printif
from termination.algorithm.utils import generate_prime_names


def toz3(c):
    if isinstance(c, list):
        return [toz3(e) for e in c]
    return c.get(Real, int, ignore_zero=True)


def generateNpoints(s, N, vs):
    s.push()
    ps = []
    rvs = [Real(v) for v in vs]
    while s.check() == sat:
        m = s.model()
        p = [float(m[v].numerator() / m[v].denominator()) for v in rvs]
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
        printif(1, "--> with " + self.NAME)
        global_vars = cfg.get_info("global_vars")
        Nvars = int(len(global_vars) / 2)
        vs = global_vars[:Nvars]
        pvs = global_vars[Nvars:]
        phi = {n: v for n, v in cfg.nodes.data("fancy_prop", default=[])}
        queue = [n for n in phi if len(phi[n]) > 0]

        while len(queue) > 0:
            node = queue.pop()
            good_cons = toz3(phi[node][:])  # phi[node] ^ ti ^ phi[ti[target]]  ^ ...
            bad_cons = []  # phi[node] ^ ti ^ ¬ phi[ti[target]] V ...
            taken_vars = []
            for t in cfg.get_edges(source=node):
                print(t["name"])
                lvs = t["local_vars"]
                taken_vars += lvs
                newprimevars = generate_prime_names(vs, taken_vars)
                taken_vars += newprimevars
                good_cons += toz3(t["polyhedron"].get_constraints(vs + newprimevars + lvs))
                good_cons += [toz3(c.renamed(vs, newprimevars)) for c in phi[t["target"]]]
                bad_tr_cons = phi[node][:] + t["polyhedron"].get_constraints() + [c.negate().renamed(vs, pvs) for c in phi[t["target"]]]
                bad_cons.append(toz3(bad_tr_cons))
                print(good_cons)
                print(bad_cons)
            Sb = Solver()
            Sb.add(Or([And(G) for G in bad_cons]))
            if Sb.check() == sat:
                bad_points = generateNpoints(Sb, 10, vs)
                Sg = Solver()
                Sg.add(good_cons)
                if Sg.check() == sat:
                    good_points = generateNpoints(Sg, 10, vs)
                else:
                    print("...")
            else:
                print("UNSAT")
            # print((solver.model()))
            # goods = solver.get
            # solver.pop()
            # solver.pop()
            # solver.pop()
        return {"status": "?"}
        print("RUNING!!")
