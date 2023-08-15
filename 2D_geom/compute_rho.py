import numpy as np
import json
import matplotlib.pyplot as pl
import matplotlib

def growth(t, t0, R, a, beta=4):
   return R/(1+np.exp(-a*(t-t0)+beta))    

def mono_sync(L,W,R):
   N_X=L/(2*R)
   N_Y = W/(np.sqrt(3)*R)
   return N_X*N_Y

def mono_desync(L, W, R, R_12):
   d = 2*(R+R_12)/np.sqrt(2)
   N_X=L/(d)
   N_Y = W/(.5*d)
   return N_X*N_Y

def mix_sync(L, W, Rl, Rc):
   d = 2*(Rc+Rl)/np.sqrt(2)
   N_X=L/(d)
   N_Y = W/(.5*d)
   return N_X*N_Y

def mix_desync(L, W, Rl, Rc, Rc_12, Rl_12):
   d_a = 2*(Rc+Rl_12)/np.sqrt(2)
   d_b = 2*(Rl+Rc_12)/np.sqrt(2)
   
   d=max(d_a, d_b)
   N_X= L/(d)
   N_Y = W/(.5*d)

   return N_X*N_Y

def split_sync(L, W, Rl, Rc):
   dc_x = 2*Rc
   dc_y = np.sqrt(3)*Rc
   dl_x = 2*Rl
   dl_y = np.sqrt(3)*Rl

   N_Y_l = L/(dc_x*dc_y/dl_x+dl_y)
   N_Y_c = N_Y_l*dc_x/dl_x

   N_X_c = L/dc_x
   N_X_l = L/dl_x
   return N_X_c*N_Y_c+N_X_l*N_Y_l

def split_desync(L, W, Rl, Rc, Rc_12, Rl_12):
   dc_x = 2*(Rc+Rc_12)/np.sqrt(2)
   dc_y = dc_x/2
   dl_x = 2*(Rl+Rl_12)/np.sqrt(2)
   dl_y = dl_x/2

   N_Y_l = L/(dc_x*dc_y/dl_x+dl_y)
   N_Y_c = N_Y_l*dc_x/dl_x

   N_X_c = L/dc_x
   N_X_l = L/dl_x
   return N_X_c*N_Y_c+N_X_l*N_Y_l

def print_rhos():
   Rmc=.8
   ac=.4
   Rml=.3
   al=.3
   t=30
   L=20
   W = 20

   Rc = growth(30, 0, Rmc, ac)
   Rc_12 = growth(15, 0, Rmc, ac)
   Rl = growth(30, 0, Rml, al)
   Rl_12 = growth(15, 0, Rml, al)

   print("Monoculture\n =========")
   N_l = mono_sync(L, W, Rl)
   N_c = mono_sync(L, W, Rc)
   print("Sync-l:", N_l/t, "c:", N_c/t)

   N_l = mono_desync(L, W, Rl, Rl_12)
   N_c = mono_desync(L, W, Rc, Rc_12)
   print("Desync-l:", N_l/t, "c:", N_c/t)

   print("Mix \n =========")
   Nms=mix_sync(L, W, Rl, Rc)
   Nmd=mix_desync(L, W, Rl, Rc, Rc_12, Rl_12)
   print("Sync:", Nms/t, "Desync", Nmd/t)

   print("Split \n =========")
   Nss = split_sync(L, W, Rl, Rc)
   Nsd = split_desync(L, W, Rl, Rc, Rc_12, Rl_12)
   print("Sync:", Nss/t, "Desync", Nsd/t)

print_rhos()

t=30
L=20
W = 20

K=50
Rms = np.linspace(.1,1,K)
a = np.linspace(.1,10,K)
beta = np.linspace(-10,10,K)
sd=np.zeros([K,K])
alphas=np.linspace(0,1,K)

#beta=4

for i in range(K):
   for j in range(K):
      #R = growth(30, 0, .5, a[j], beta[i]) 
      #R_12 = growth(15, 0, .5, a[j], beta[i])
      R = Rms[i] 
      R_12 = alphas[j]*Rms[i]

      N_s=mono_sync(L, W, R)
      N_d = mono_desync(L, W, R, R_12)
      sd[i,j] = (N_s-N_d)/(N_s+N_d) 