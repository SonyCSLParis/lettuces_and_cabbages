import intercropsim as sim
import utils as ut
import json
import time
import shutil
from joblib import Parallel, delayed
import os
import numpy as np

def one_run(pr = 1, name = "test", svg_folder = "res/test", plot_folder = "figs/"):
   t0 = time.time()
   if not(os.path.exists(svg_folder)): os.mkdir(svg_folder)
   sim_params = json.load(open('default.json'))
   sim_params["planting_rate"] = pr  
   sim_params["N"] = 4000
   ts = sim.sim(sim_params, svg_folder+name+".json")
   sim_data = json.load(open(svg_folder+name+".json"))
   ut.plot_field(sim_data["plants"], sim_data["sim_params"], [0,sim_data["plants"][-1]["t"]], plot_folder+name+".png")
   ut.plot_field(sim_data["plants"], sim_data["sim_params"], [0,4000], plot_folder+name+".png")
   print(time.time()-t0, "s for pr = %.2f"%pr)
   return [pr,ts]

res = one_run(.45, name = "test")
