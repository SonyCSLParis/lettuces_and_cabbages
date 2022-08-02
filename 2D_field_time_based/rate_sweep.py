import intercropsim as sim
import numpy as np

pc =.5
prs = np.linspace(0.05, 5, 50)
ms = []
params = json.load(open("default.json"))

for pr in prs:
   m = sim(pr, pc, 2000, params['R'], params['a'])
   ms.append(m)
np.savetxt("data/rate_var/meas.txt", ms)
np.savetxt("data/rate_var/rates.txt", prs)
