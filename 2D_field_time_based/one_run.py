import intercropsim as sim
import json
import time

t0 = time.time()
N = 2000
params = json.load(open('default.json'))
all_ts, sel_ts = sim.sim(3, 0.5, N, params['R'], params['a'], params["tmax"], "test.json")
print(time.time()-t0)
