from z3 import Solver
from z3 import Real
from z3 import substitute
from z3 import sat
from main import printif
from genericparser.expressions import ExprTerm


class ML:
    ID = "ml"
    NAME = "ml"
    DESC = "non termination using ML"

    def __init__(self, properties={}):
        self.props = properties

    def project(self, cons, global_vars, rigth=False):
        from ppl import Constraint_System
        from lpi import C_Polyhedron
        Nvars = int(len(global_vars) / 2)
        tr_poly = C_Polyhedron(Constraint_System([c.transform(global_vars, lib="ppl") for c in cons]), dim=len(global_vars))
        # PROJECT TO VARS
        from ppl import Variables_Set
        from ppl import Variable
        var_set = Variables_Set()
        _vars = []
        offset = 0
        if not rigth:
            # original variables
            for i in range(0, Nvars):
                var_set.insert(Variable(i))
                _vars.append(Variable(i+Nvars))
        else:
            # prime variables
            for i in range(Nvars, 2*Nvars):
                var_set.insert(Variable(i))
                _vars.append(Variable(i-Nvars))
            offset = Nvars
        # local variables
        for i in range(2*Nvars, tr_poly.get_dimension()):  # Vars from n to m-1 inclusive
            var_set.insert(Variable(i))
        print(tr_poly)
        tr_poly.remove_dimensions(var_set)
        final_cons = []
        for c_ppl in tr_poly.get_constraints():
            equation = ExprTerm(c_ppl.inhomogeneous_term())
            for i in range(0, Nvars):
                cf = c_ppl.coefficient(Variable(i))
                if cf != 0:
                    equation += ExprTerm(cf) * ExprTerm(global_vars[i+offset])
                if c_ppl.is_equality():
                    final_cons.append(equation == ExprTerm(0))
                else:
                    final_cons.append(equation >= ExprTerm(0))
        return final_cons

    def toz3(self, cons, global_vars):
        return [c.transform(global_vars, lib="z3") for c in cons]


    def run(self, cfg):
        from z3 import Not, And, simplify
        printif(1, "--> with "+self.NAME)
        global_vars = cfg.get_info("global_vars")
        Nvars = int(len(global_vars) / 2)
        vs = global_vars[:Nvars]
        pvs = global_vars[Nvars:]
        phi = {n:v for n, v in cfg.nodes.data("fancy_prop", default=[])}
        trs_cons = []
        for t in cfg.get_edges():
            # Creating Solver per transition
            s = self.toz3(t["constraints"], global_vars+t["local_vars"])
            trs_cons.append((t["name"], t["source"], t["target"], s))
            
        print(phi)
        stop = False
        solver = Solver()
        while(not stop):
            for name, src, trg, s in trs_cons:
                print(name, src, trg, s)
                solver.push()
                solver.add(s)
                solver.push()
                # adding PHI(src)
                solver.add(self.toz3(self.project(phi[src], global_vars), global_vars))
                
                solver.push() # pushing later I will do just the oposite here
                solver.add(Not(And(self.toz3(self.project(phi[trg], global_vars, rigth=True), global_vars))))
            if solver.check() != sat:
                
                print("UNSAT!!!")
                stop = True
                break
            print((solver.model()))
            #goods = solver.get
            solver.pop()
            solver.pop()
            solver.pop()
        return {"status":"?"}
        print("RUNING!!")