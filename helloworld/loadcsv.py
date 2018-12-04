import argparse
import sys
import csv

_version = "0.0.0"
_name = "CFGstats"

def setArgumentParser():
    desc = _name+": CFG stats generator to CSV."
    argParser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.RawTextHelpFormatter)
    # Program Parameters
    argParser.add_argument("-v", "--verbosity", type=int, choices=range(0, 5),
                           help="increase output verbosity", default=0)
    # IMPORTANT PARAMETERS
    argParser.add_argument("-f", "--files", nargs='+', required=True,
                           help="File to be analysed.")
    from time import gmtime, strftime
    defaultdest = "data-"+strftime("%Y%m%d%H%M%S", gmtime())+".csv"
    argParser.add_argument("-d", "--destination", required=False, default=defaultdest,
                           help="Temporary directory.")
    argParser.add_argument("-pt", "--pe_times", type=int, choices=range(0, 5),
                           help="# times to apply pe", default=0)
    return argParser

def get_stats_cfg(cfg):
    # Objects
    trs = cfg.get_edges()
    list_cons = [len(t["constraints"]) for t in trs]
    list_sccs = cfg.get_scc()
    list_sccs_ntrs = [len(scc.get_edges()) for scc in list_sccs if len(scc.get_edges()) > 0]
    
    # Stats
    Ntrs = len(trs)
    Ncons = sum(list_cons)
    MaxStr = max(list_cons)
    AvgStr = Ncons/Ntrs
    Nsccs = len(list_sccs_ntrs)
    MaxSscc = max(list_sccs_ntrs)
    AvgSscc = sum(list_sccs_ntrs)/Nsccs
    
    return Ntrs, Ncons, MaxStr, AvgStr, Nsccs, MaxSscc, AvgSscc


def get_stats_file(file, pe_times=0):
    import genericparser
    from partialevaluation import partialevaluate
    cfg = genericparser.parse(file)
    pe_cfgs = []
    pe_cfgs.append(cfg)
    for i in range(1,pe_times+1):
        pe_cfgs.append(partialevaluate(pe_cfgs[i-1]))
    stats = {}
    stats["filename"] = file
    prefix=""
    for i in range(pe_times+1):
        Ntrs, Ncons, MaxStr, AvgStr, Nsccs, MaxSscc, AvgSscc = get_stats_cfg(pe_cfgs[i])
        # Write dict
        stats[prefix+"Ncons"] = Ncons
        stats[prefix+"Ntrs"] = Ntrs
        stats[prefix+"MaxStr"] = MaxStr
        stats[prefix+"AvgStr"] = AvgStr
        stats[prefix+"Nsccs"] = Nsccs
        stats[prefix+"MaxSscc"] = MaxSscc
        stats[prefix+"AvgSscc"] = AvgSscc
        prefix = "PE"+str(i+1)
    return stats
    
def launch(config):
    csvdest = config["destination"]
    keys = ["filename", "Ncons", "Ntrs", "MaxStr", "AvgStr", "Nsccs", "MaxSscc", "AvgSscc" ]
    with open(csvdest, 'w') as csvfile:
        statswrt = csv.DictWriter(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, fieldnames=keys)
        statswrt.writeheader()
        count = 0
        mx = len(config["files"])
        for f in config["files"]:
            count = count + 1
            print("({}/{}) {}".format(count,mx,f))
            stats = get_stats_file(f, pe_times=config["pe_times"])
            try:
                pass
            except Exception as e:
                print(str(f) + " raises: " + str(e))
                continue

            statswrt.writerow(stats)
            

if __name__ == "__main__":
    argv = sys.argv[1:]
    argParser = setArgumentParser()
    args = argParser.parse_args(argv)
    config = vars(args)
    launch(config)
