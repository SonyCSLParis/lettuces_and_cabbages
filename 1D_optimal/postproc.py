import matplotlib.pyplot as pl
import numpy as np
import json
import glob

def get_params(folder):
   files = glob.glob(folder+"*")
   files.sort()
   f = files[0]
   data = json.load(open(f))   
   return data["sim_params"]


def get_meas(folder):
   res = {}

   files = glob.glob(folder+"*")
   files.sort()

   for f in files:
      pr = float(f.split("/")[-1].split("_")[0])

      if not(pr in res.keys()):
         res[pr] = {"Nc"       : [],
                    "Nl"       : [],
                    "r_c"      : [],
                    "st"       : [],
                    "tmax"     : [],
                    "N_trials" : [],
                    "dts"      : [],
                    "density_c": [],
                    "density_l": []
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
      res[pr]["dts"].append([np.mean(np.diff(data["sel_ts"]))])
      res[pr]["density_c"].append([np.mean(Nc/(data["plants"][-1]["t"]+30))])
      res[pr]["density_l"].append([np.mean(Nl/(data["plants"][-1]["t"]+30))])
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

   density_c_m = [np.mean(res[r]["density_c"]) for r in res]
   density_c_v = [np.std(res[r]["density_c"]) for r in res]

   density_l_m = [np.mean(res[r]["density_l"]) for r in res]
   density_l_v = [np.std(res[r]["density_l"]) for r in res]

   dts_m = [np.mean(res[r]["dts"]) for r in res]
   dts_v = [np.std(res[r]["dts"]) for r in res]

   Nt_m = np.zeros(len(res))
   Nt_v = np.zeros(len(res))

   for i, r in enumerate(res):
      Nt = np.array(res[r]["N_trials"]) 
      Nt_m[i] = np.mean(Nt[:,0])
      Nt_v[i] = np.std(Nt[:,0])
      
   return {"Nc mean": list(Ncs_m), "Nc std": list(Ncs_v), 
           "Nl mean": list(Nls_m), "Nl std": list(Nls_v), 
           "ratio c/l mean":list(rcs_m), "ratio c/l std": list(rcs_v), 
           "sim time mean": list(st_m), "sim time std": list(st_v), 
           "tmax mean": list(tmax_m), "tmax std": list(tmax_v), 
           "N trials mean": list(Nt_m), "N trials std": list(Nt_v), 
           "dts mean": list(dts_m), "dts std": list(dts_v),
           "density_c mean": list(density_c_m), "density_c std": list(density_c_m), 
           "density_l mean": list(density_l_m), "density_l std": list(density_l_m), 
           "prs": list(prs)
           }

def fig_Nc(s):
   pl.plot(s["prs"], s["Nc mean"], "--", color = "k")
   #pl.errorbar(s["prs"], s["Nc mean"], yerr = s["Nc std"], fmt="-", color = "k")
   pl.xlabel("Planting rate")
   pl.ylabel("# cabbages")
   pl.savefig("figs/Nc.png", bbox_inches = "tight")
   pl.clf()

def fig_dts(s):
   pl.plot(s["prs"], 1/np.array(s["dts mean"]), "-", color = "k")
   #pl.errorbar(s["prs"], s["dts mean"], yerr = s["dts std"], fmt="o--", color = "k")
   pl.xlabel("Planting rate")
   pl.ylabel("Effective rate")
   pl.savefig("figs/eff_rate.png", bbox_inches = "tight")
   pl.clf()

def fig_density(s):
   #p = get_params(folder)
   #N = p["N"]
   N=4000
   pl.plot(s["prs"],N/np.array(s["tmax mean"]), "k")
   pl.xlabel("Planting rate")
   pl.ylabel("Density")
   pl.savefig("figs/density.png", bbox_inches = "tight")
   pl.clf()
   
def fig_density_lc(s):
   #p = get_params(folder)
   #N = p["N"]
   N=4000
   Nc = s["Nc mean"]
   pl.fill_between(s["prs"], np.zeros(len(s["density_c mean"])), s["density_c mean"], color = [0.1, 0.4, 0.1])
   pl.fill_between(s["prs"], s["density_c mean"], np.array(s["density_c mean"])+np.array(s["density_l mean"]), color = [0.1, 0.8, 0.1])

   #pl.plot(stats["prs"],N/np.array(stats["tmax mean"]))
   pl.xlabel("Planting rate")
   pl.ylabel("Density")
   pl.savefig("figs/density_lc.png", bbox_inches = "tight")
   pl.clf()

def fig_ratio(s):
   pl.fill_between(s["prs"], np.zeros(len(s["ratio c/l mean"])), s["ratio c/l mean"], color = [0.1, 0.4, 0.1])
   pl.fill_between(s["prs"], s["ratio c/l mean"], np.ones(len(s["ratio c/l mean"])), color = [0.1, 0.8, 0.1])
   pl.xlabel("Planting rate")
   pl.ylabel("cabbages/lettuces")
   pl.savefig("figs/ratio.png")
   pl.clf()

def fig_sim_time(s):
   var = "N trials mean"
   #pl.errorbar(s["prs"], s["N trials mean"], yerr = s["N trials std"], fmt="-", color = "k")
   pl.plot(s["prs"], s["N trials mean"], color = "k")
   pl.xlabel("Planting rate")
   pl.ylabel("N trails")
   pl.savefig("figs/N_trials.png")
   pl.clf()

   #pl.errorbar(s["prs"], s["sim time mean"], yerr = s["sim time std"], fmt="-", color = "k")
   pl.plot(s["prs"], s["sim time mean"],  color = "k")
   pl.xlabel("Planting rate")
   pl.ylabel("sim time")
   pl.savefig("figs/sim_time.png")
   pl.clf()

s=json.load(open("res/stats_lettuces.json"))

fig_Nc(s)
fig_density(s)
fig_density_lc(s)
fig_ratio(s)
fig_sim_time(s)
fig_dts(s)



#folder = "/home/kodda/Dropbox/p2pflab/lettuces_and_cabbages/1D_optimal/res/prs_force_proba/"
#folder = "/home/kodda/Dropbox/p2pflab/lettuces_and_cabbages/1D_optimal/res/prs/"

"""
meas = get_meas(folder)
s = get_stats(meas)

json.dump(s, open("res/stats_sel.json","w"))

folder = "/home/kodda/Dropbox/p2pflab/lettuces_and_cabbages/1D_optimal/res/prs_force_proba/"
meas = get_meas(folder)
s = get_stats(meas)

json.dump(s, open("res/stats_force_proba.json","w"))
"""

"""
p = get_params(folder)
N = p["N"]
Nc = s["Nc mean"]

#pl.plot(stats["prs"],N/np.array(stats["tmax mean"]))

files = glob.glob(folder+"*")
files.sort()
f = files[-1]
data = json.load(open(f))   



fig_Nc(s)
fig_density(s, folder)
fig_density_lc(s, folder)
fig_ratio(s)
fig_sim_time(s)
fig_dts(s)


files = glob.glob(folder+"*")
files.sort()
f = files[0]
data = json.load(open(f))   
"""