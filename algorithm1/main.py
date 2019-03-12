'''
Created on Dec 4, 2018

@author: Jesus Doménech
@email jdomenec@ucm.es
'''
import argparse
import os
import sys
from termination.output import Output_Manager as OM

_name = "alg1"
_version = "v0.0.0dev"


def setArgumentParser():
    desc = _name + ": test script for Non-Termination algorithm with ML."
    argParser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.RawTextHelpFormatter)
    # Program Parameters
    argParser.add_argument("-v", "--verbosity", type=int, choices=range(0, 5),
                           help="increase output verbosity", default=0)
    argParser.add_argument("-V", "--version", required=False,
                           action='store_true', help="Shows the version.")
    argParser.add_argument("--ei-out", required=False, action='store_true',
                           help="Shows the output supporting ei")
    argParser.add_argument("-sccd", "--scc-depth", type=int, default=4,
                           help="Strategy based on SCC to go through the CFG.")
    # ntML parameters
    argParser.add_argument("-p", "--plot", required=False, action='store_true', help="Plot splitting if number vars is 2")
    argParser.add_argument("-s", "--sampling", default="solver", choices=["solver", "lp", "rays"],
                           help="Sampling method")
    argParser.add_argument("-sn", "--sampling-number", type=int, default=10,
                           help="Number of points.")
    argParser.add_argument("-st", "--sampling-threshold", type=int, default=2,
                           help="Half of the minimum distance between two points in sampling.")
    argParser.add_argument("-c", "--classifier", default="linear", choices=["linear", "svc", "perceptron"],
                           help="Classifier to be used.")
    argParser.add_argument("-ptr", "--phi-transition", required=False, action='store_true',
                           help="One Phi per node and transition.")
    argParser.add_argument("-pnd", "--phi-nondeterminism", required=False, action='store_true',
                           help="Phi splitting no-determinism.")
# IMPORTANT PARAMETERS
    argParser.add_argument("-f", "--files", nargs='+', required=True,
                           help="File to be analysed.")
    return argParser


def extractname(filename):
    f = os.path.split(filename)
    b = os.path.split(f[0])
    c = os.path.splitext(f[1])
    return os.path.join(b[1], c[0])


def launch(config):
    files = config["files"]
    for i in range(len(files)):
        if(len(files) > 1):
            print(files[i])
        config["name"] = extractname(files[i])
        launch_file(config, files[i])


def parse_file(f):
    import genericparser
    return genericparser.parse(f)


def launch_file(config, f):
    try:
        cfg = parse_file(f)
    except Exception as e:
        print("Parser Error: {}\n{}".format(type(e).__name__, str(e)))
        raise Exception() from e
        return False
    config["vars_name"] = cfg.get_info("global_vars")
    from nodeproperties import invariant
    invariant.set_configuration({"invariants": "polyhedra", "invariants_threshold": []})
    invariant.compute_invariants(cfg, add_to_polyhedron=True)
    analyse(config, cfg)
    return True


def analyse(config, cfg):
    max_sccd = config["scc_depth"]
    CFGs = [(cfg, max_sccd)]
    stop = False
    maybe_sccs = []
    terminating_sccs = []
    nonterminating_sccs = []
    ntMLkeys = ["plot", "sampling", "sampling_number", "sampling_threshold", "classifier", "phi_transition", "phi_nondeterminism"]
    ntMLconf = {k: config[k] for k in ntMLkeys}
    OM.printif(2, "config: {")
    for key in sorted(ntMLconf.keys()):
        OM.printif(2, "\t%s: %s" % (key, ntMLconf[key]))
    OM.printif(2, "}")
    while (not stop and CFGs):
        current_cfg, sccd = CFGs.pop(0)
        removed = current_cfg.remove_unsat_edges()
        if len(removed) > 0:
            OM.printif(2, "Transition (" + removed + ") were removed because it is empty.")
        if len(current_cfg.get_edges()) == 0:
            OM.printif(2, "This cfg has not transitions.")
            continue
        cfg_cfr = current_cfg
        CFGs_aux = cfg_cfr.get_scc() if sccd > 0 else [cfg_cfr]
        sccd -= 1
        CFGs_aux.sort()

        for scc in CFGs_aux:
            scc.remove_unsat_edges()
            if len(scc.get_edges()) == 0:
                continue
            cons, nivars = scc.remove_no_important_variables()
            OM.printif(1, "Removed {} constraint(s) with variable(s) for one scc: [{}]".format(cons, ",".join(nivars)))
            R = analyse_scc(scc, ntMLconf)
            if R.get_status().is_terminate():
                terminating_sccs.append(R)
            elif R.get_status().is_nonterminate():
                nonterminating_sccs.append(R)
            else:
                maybe_sccs.append(R)
        # end for
    # end while
    status = "maybe"
    if len(nonterminating_sccs) > 0:
        status = "no"
    elif len(maybe_sccs) > 0:
        status = "maybe"
    else:
        status = "yes"
    response = {"status": status,
                "terminate": terminating_sccs,
                "nonterminate": nonterminating_sccs,
                "unknown_sccs": maybe_sccs,
                "graph": cfg}
    return response


def analyse_scc(scc, config):
    from ntML import ML
    alg = ML()
    return alg.run(scc, config)


if __name__ == '__main__':
    try:
        argv = sys.argv[1:]
        argParser = setArgumentParser()
        args = argParser.parse_args(argv)
        if args.version:
            print(_name + " version: " + _version)
            exit(0)
        OM.restart(verbosity=args.verbosity, ei=args.ei_out)
        config = vars(args)
        if "verbosity" in config:
            verbosity = config["verbosity"]
        launch(config)
    except Exception as e:
        print("Error: {}\n{}".format(type(e).__name__, str(e)))
        raise Exception() from e
