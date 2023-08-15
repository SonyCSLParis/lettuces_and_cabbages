import numpy as np
import json
import matplotlib.pyplot as pl
import matplotlib

def growth(t, t0, R, a):
   return R/(1+np.exp(-a*(t-t0)+4))    

Rmc=.8
ac=.4
Rml=.3
al=.3
t=30
L=100

Rc = growth(30, 0, Rmc, ac)
Rc_12 = growth(15, 0, Rmc, ac)
Rl = growth(30, 0, Rml, al)
Rl_12 = growth(15, 0, Rml, al)

t1=1
msl = L/(2*Rl*t)
msc = L/(2*Rc*t)
mdl = L/((Rl+Rl_12)*t)
mdc = L/((Rc+Rc_12)*t)
print("Monoculture\n =========")
print("Sync-l:", msl*t1, "c:", msc*t1)
print("Desync-l:", mdl*t1, "c:", mdc*t1)

print("Mix sync \n =========")
mix_sync = L/((Rl+Rc)*t)
print(mix_sync*t1)

print("Split desync \n =========")
split_desync = 2*L/((Rc+Rc_12+Rl+Rl_12)*t)
print(split_desync*t1)

print("Intercrop desync \n =========")
inter_desync = L/((max(Rc+Rl_12, Rl+Rc_12))*t)
print(inter_desync*t1)
