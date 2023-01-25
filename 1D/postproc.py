import matplotlib.pyplot as pl
import numpy as np
import json
import glob

def get_meas(folder, prefix):
   res = {}

   files = glob.glob(folder+prefix+"*")
   files.sort()

   for f in files:
      fsplit = f.split("/")[-1][:-5].split("_")
      pr = float(fsplit[-2])
      print(pr)
      if not(pr in res.keys()):
         res[pr] = {"Nc"       : [],
                    "Nl"       : [],
                    "r_c"      : [],
                    "st"       : [],
                    "tmax"     : [],
                    "N_trials" : [],
                    "dts"      : []
                   }   

      data = json.load(open(f))   

      sps = np.array([(p["species"]=="c") for p in data["plants"]])
      Nc = sps.sum()
      Nl = (1-sps).sum()
      res[pr]["Nc"].append(Nc) 
      res[pr]["Nl"].append(Nl)
      res[pr]["r_c"].append(Nc/(Nc+Nl))
      res[pr]["st"].append(data["sim_time"])
      res[pr]["tmax"].append(data["plants"][-1]["t"]+30)
      res[pr]["N_trials"].append([np.mean(data["ms"]), np.var(data["ms"])])
      res[pr]["dts"].append([np.mean(data["sel_ts"]), np.var(data["sel_ts"])])
   res = dict(sorted(res.items()))
   return res

def get_stats(res):
   prs = list(res.keys())
   Ncs_m = [np.mean(res[r]["Nc"]) for r in res]
   Ncs_v = [np.std(res[r]["Nc"]) for r in res]

   Nls_m = [np.mean(res[r]["Nl"]) for r in res]
   Nls_v = [np.std(res[r]["Nl"]) for r in res]

   rcs_m = [np.mean(res[r]["r_c"]) for r in res]
   rcs_v = [np.std(res[r]["r_c"]) for r in res]

   st_m = [np.median(res[r]["st"]) for r in res]
   st_v = [np.std(res[r]["st"]) for r in res]

   tmax_m = [np.mean(res[r]["tmax"]) for r in res]
   tmax_v = [np.std(res[r]["tmax"]) for r in res]

   Nt_m = np.zeros(len(res))
   Nt_v = np.zeros(len(res))
   dts_m = np.zeros(len(res))
   dts_v = np.zeros(len(res))

   for i, r in enumerate(res):
      Nt = np.array(res[r]["N_trials"]) 
      Nt_m[i] = np.median(Nt[:,0])
      Nt_v[i] = np.sqrt(Nt[:,1].mean())
      dts = np.array(res[r]["dts"]) 
      dts_m[i] = dts[:,0].mean()
      dts_v[i] = np.sqrt(dts[:,1].mean())
   return {"Nc mean": list(Ncs_m), "Nc std": list(Ncs_v), 
           "Nl mean": list(Nls_m), "Nl std": list(Nls_v), 
           "ratio c/l mean":list(rcs_m), "ratio c/l std": list(rcs_v), 
           "sim time mean": list(st_m), "sim time std": list(st_v), 
           "tmax mean": list(tmax_m), "tmax std": list(tmax_v), 
           "N trials mean": list(Nt_m), "N trials std": list(Nt_v), 
           "dts mean": list(dts_m), "dts std": list(dts_v), "prs": list(prs)}

def fig_Nc(stats, waste = False):
   s = stats["smart_False_waste_%s"%waste]
   pl.plot(s["prs"], s["Nc mean"], "-", color = "k", label="Naive")
   s = stats["smart_True_waste_%s"%waste]
   pl.plot(s["prs"], s["Nc mean"], "-", color = "r", label="Smart")
   pl.xlabel("Planting rate")
   pl.ylabel("# cabbages")
   pl.title("Waste %s"%waste)
   pl.legend()
   pl.savefig("figs/Nc_%s.png"%waste, bbox_inches = "tight")
   pl.clf()

def fig_ratio(stats, waste = False):
   s = stats["smart_False_waste_%s"%waste]
   pl.fill_between(s["prs"], np.zeros(len(s["ratio c/l mean"])), s["ratio c/l mean"], color = [0.1, 0.4, 0.1])
   pl.fill_between(s["prs"], s["ratio c/l mean"], np.ones(len(s["ratio c/l mean"])), color = [0.1, 0.8, 0.1])
   pl.xlabel("Planting rate")
   pl.ylabel("cabbages/lettuces")
   pl.title("Waste %s"%waste)
   pl.savefig("figs/ratio_%s.png"%waste)
   pl.clf()

def fig_sim_time(stats, waste):
   var = "N trials mean"
   s = stats["smart_False_waste_%s"%waste]
   #pl.errorbar(s["prs"], s["N trials mean"], yerr = s["N trials std"], fmt="o-", color = "k", label = "Naive")
   pl.plot(s["prs"], np.array(s[var]), "o-", color = "k", label = "Naive")

   s = stats["smart_True_waste_%s"%waste]
   #pl.errorbar(s["prs"], s["N trials mean"], yerr = s["N trials std"], fmt="o-", color = "r", label="Smart")
   pl.plot(s["prs"], np.array(s[var]), "o-", color = "r", label = "Smart")
   pl.xlabel("Planting rate")
   pl.ylabel("N trials")
   pl.title("Waste %s"%waste)
   pl.savefig("figs/N_trials_%s.png"%waste)
   pl.clf()


folder = "/home/kodda/Dropbox/p2pflab/lettuces_and_cabbages/1D/res/"

"""
prefixs = ["smart_False_waste_False", "smart_False_waste_True", "smart_True_waste_False", "smart_True_waste_True"]
#prefixs = ["smart_False_waste_False", "smart_True_waste_False"]

stats = {}

for prefix in prefixs:
   meas = get_meas(folder, prefix)
   stats[prefix] = get_stats(meas)
"""
#json.dump(stats, open("stats.json","w"))

stats = json.load(open("stats.json"))

#fig_Nc(stats, waste = False)
#fig_Nc(stats, waste = True)

#fig_ratio(stats, waste = False)
#fig_ratio(stats, waste = True)

fig_sim_time(stats, True)
fig_sim_time(stats, False)


"""
ds = [json.load(open(f)) for f in files]

prs = np.array([d["sim_params"]["planting_rate"] for d in ds])
idxs = np.argsort(prs)
ds = [ds[idx] for idx in idxs]
prs = prs[idxs]

Ncs = np.zeros(len(ds))
Nls = np.zeros(len(ds))

for i in range(len(ds)):
   sps = np.array([(p["species"]=="c") for p in ds[i]["plants"]])
   Ncs[i] = sps.sum() 
   Nls[i] = (1-sps).sum()

pl.plot(prs,Ncs, "k") 
pl.xlabel("Planting rate")
pl.ylabel("# cabbages")
pl.savefig("figs/1D_EB.png")
"""