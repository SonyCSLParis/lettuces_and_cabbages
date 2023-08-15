import numpy as np
import json
import matplotlib.pyplot as pl

def growth(t, t0, R, a):
   return R/(1+np.exp(-a*(t-t0)+4))

def plot_field(t, plants, params, svg, title=None):
   fig=pl.figure(figsize=(10,10))
   ax = fig.add_subplot(111)

   fig.patch.set_facecolor("#805b21")
   pl.axis("off")

   pl.plot([0,0,20,20],[0,20,0,20],".")

   for p in plants:
      if (t>p["t"])*(t<p["t"]+params["tmax"][p["species"]]): 
         r = growth(t, p["t"], params["R"][p["species"]], params["a"][p["species"]])
         circle = pl.Circle([p["x"], p["y"]], r, color=params["cols"][p["species"]])
         ax.add_artist(circle)
   if title: pl.title(title)
   pl.savefig(svg, facecolor=fig.get_facecolor(),bbox_inches="tight")
   pl.clf()

def get_Nx_Ny_mono_desync(L, W, species, sim_params):
   Rm = sim_params["R"][species]

   a = sim_params["a"][species] 

   R = growth(30, 0, Rm, a)
   R_12 = growth(15, 0, Rm, a)

   d = 2*(R+R_12)/np.sqrt(2)

   N_X=np.ceil(L/(d)) -1
   N_Y = np.ceil(W/(.5*d))-1
   return N_X, N_Y, d, R

def get_xy_mono_desync(L, W, species, sim_params):
   N_X, N_Y, d, R = get_Nx_Ny_mono_desync(L, W, species, sim_params)
   print("Nx",N_X,"Ny", N_Y,"N", N_X*N_Y)
   xv, yv = np.meshgrid(np.arange(N_X), np.arange(N_Y), sparse=False, indexing='xy')
   ts = 15*(yv%2)
   yv*= d/2
   yv+=R
   xv*=d
   xv[1::2, :] += d/2
   xv+=R
   return xv.flatten(),yv.flatten(), ts.flatten()

def fig_mono_desync(species):
   sim_params = json.load(open("/home/kodda/Dropbox/p2pflab/lettuces_and_cabbages/2D_geom/default.json"))
   L = 20
   W = L
   t_h = sim_params["tmax"]["c"]
   xs,ys, ts = get_xy_mono_desync(L, W, species, sim_params)

   plants = []
   for i in range(len(xs)):
      plants.append({'x': xs[i], 'y' : ys[i], 't': ts[i], 'species': species, "id": i})

   plot_field(29.5, plants, sim_params, "mono_desync_%s.png"%species, "N/t=%.2f"%(len(xs)/t_h))

def get_Nx_Ny_mono_sync(L, W, species, sim_params):
   Rm = sim_params["R"][species]
   a = sim_params["a"][species]
   R = growth(30, 0, Rm, a)

   N_X=np.ceil(L/(2*R)) -1
   N_Y = np.ceil(W/(np.sqrt(3)*R))-1
   return N_X, N_Y, R

def get_xy_mono_sync(L, W, species, sim_params):
   N_X, N_Y, R = get_Nx_Ny_mono_sync(L, W, species, sim_params)
   print("Nx",N_X,"Ny", N_Y,"N", N_X*N_Y)
   xv, yv = np.meshgrid(np.arange(N_X), np.arange(N_Y), sparse=False, indexing='xy')
   yv*= np.sqrt(3)*R
   yv+=sim_params["R"][species]
   xv*=2*R
   xv[1::2, :] += R
   xv+=R
   return xv.flatten(),yv.flatten()

def fig_mono_sync(species):
   sim_params = json.load(open("/home/kodda/Dropbox/p2pflab/lettuces_and_cabbages/2D_geom/default.json"))
   L = 20
   W = L
   t_h = sim_params["tmax"]["c"]
   xs, ys = get_xy_mono_sync(L, W, species, sim_params)

   plants = []
   for i in range(len(xs)):
      plants.append({'x': xs[i], 'y' : ys[i], 't': 0, 'species': species, "id": i})

   plot_field(29.5, plants, sim_params, "mono_sync_%s.png"%species, "N/t=%.2f"%(len(xs)/t_h))

def get_Nx_Ny_mix_desync(L, W, sim_params):
   Rml = sim_params["R"]["l"]
   al = sim_params["a"]["l"] 
   Rmc = sim_params["R"]["c"]
   ac = sim_params["a"]["c"] 

   Rc = growth(30, 0, Rmc, ac)
   Rc_12 = growth(15, 0, Rmc, ac)
   Rl = growth(30, 0, Rml, al)
   Rl_12 = growth(15, 0, Rml, al)

   d_a = (Rc+Rl_12)/np.sqrt(2)
   d_b = (Rl+Rc_12)/np.sqrt(2)
   
   d=max(d_a, d_b)

   N_X=np.ceil(L/(2*d)) -1
   N_Y = np.ceil(W/(d))-1
   return N_X, N_Y, d, Rc, Rl

def get_xy_mix_desync(L, W, sim_params):
   N_X, N_Y, d, Rc, Rl = get_Nx_Ny_mix_desync(L, W, sim_params)
   print("Nx",N_X,"Ny", N_Y,"N", N_X*N_Y)
   xv, yv = np.meshgrid(np.arange(N_X), np.arange(N_Y), sparse=False, indexing='xy')
   ts = 15*(yv%2)
   ts=15-ts
   s = np.array(['l', 'c'])
   species = s[(yv%2==0).flatten().astype("int")]
   print(d)
   yv*= d
   yv+=Rc
   xv*=2*d
   xv[1::2, :] += d
   xv+=Rc
   return xv.flatten(),yv.flatten(), ts.flatten(), species

def fig_mix_desync():
   sim_params = json.load(open("/home/kodda/Dropbox/p2pflab/lettuces_and_cabbages/2D_geom/default.json"))
   L = 20
   W = L
   t_h = sim_params["tmax"]["c"]
   xs,ys, ts, species = get_xy_mix_desync(L, W, sim_params)

   plants = []
   for i in range(len(xs)):
      plants.append({'x': xs[i], 'y' : ys[i], 't': ts[i], 'species': species[i], "id": i})

   plot_field(29.5, plants, sim_params, "mix_desync.png", "N/t=%.2f"%(len(xs)/t_h))

def get_Nx_Ny_mix_sync(L, W, sim_params):
   Rml = sim_params["R"]["l"]
   al = sim_params["a"]["l"] 
   Rmc = sim_params["R"]["c"]
   ac = sim_params["a"]["c"] 

   Rc = growth(30, 0, Rmc, ac)
   Rc_12 = growth(15, 0, Rmc, ac)
   Rl = growth(30, 0, Rml, al)
   Rl_12 = growth(15, 0, Rml, al)

   d = (Rc+Rl)/np.sqrt(2)
   
   N_X=np.ceil(L/(2*d)) -1
   N_Y = np.ceil(W/(d))-1
   return N_X, N_Y, d, Rc, Rl

def get_xy_mix_sync(L, W, sim_params):
   N_X, N_Y, d, Rc, Rl = get_Nx_Ny_mix_sync(L, W, sim_params)
   print("Nx",N_X,"Ny", N_Y,"N", N_X*N_Y)
   xv, yv = np.meshgrid(np.arange(N_X), np.arange(N_Y), sparse=False, indexing='xy')
   ts = 15*(yv%2)
   ts=15-ts
   s = np.array(['l', 'c'])
   species = s[(yv%2==0).flatten().astype("int")]
   print(d)
   yv*= d
   yv+=Rc
   xv*=2*d
   xv[1::2, :] += d
   xv+=Rc
   return xv.flatten(),yv.flatten(), ts.flatten(), species

def fig_mix_sync():
   sim_params = json.load(open("/home/kodda/Dropbox/p2pflab/lettuces_and_cabbages/2D_geom/default.json"))
   L = 20
   W = L
   t_h = sim_params["tmax"]["c"]
   xs,ys, ts, species = get_xy_mix_sync(L, W, sim_params)

   plants = []
   for i in range(len(xs)):
      plants.append({'x': xs[i], 'y' : ys[i], 't': ts[i], 'species': species[i], "id": i})

   plot_field(29.5, plants, sim_params, "mix_sync.png", "N/t=%.2f"%(len(xs)/t_h))


def get_Nx_Ny_split_sync(L, W, sim_params):
   Rml = sim_params["R"]["l"]
   al = sim_params["a"]["l"] 
   Rmc = sim_params["R"]["c"]
   ac = sim_params["a"]["c"] 

   Rc = growth(30, 0, Rmc, ac)
   Rl = growth(30, 0, Rml, al)

   dc_x = 2*Rc
   dc_y = np.sqrt(3)*Rc
   dl_x = 2*Rl
   dl_y = np.sqrt(3)*Rl

   N_Y_l = L/(dc_x*dc_y/dl_x+dl_y)
   N_Y_c = N_Y_l*dc_x/dl_x

   N_X_c = L/dc_x
   N_X_l = L/dl_x

   return np.floor(N_X_c), np.floor(N_X_l), np.floor(N_Y_c), N_Y_l, dc_x, dc_y, dl_x, dl_y

def get_xy_split_sync(L, W, sim_params):
   N_X_c, N_X_l, N_Y_c, N_Y_l, dc_x, dc_y, dl_x, dl_y = get_Nx_Ny_split_sync(L, W, sim_params)
   print("Nc", N_X_c*N_Y_c,"Nc", N_X_c*N_Y_c)
   xv_c, yv_c = np.meshgrid(np.arange(N_X_c), np.arange(N_Y_c), sparse=False, indexing='xy')
   yv_c*= dc_y
   yv_c+=sim_params["R"]['c']
   xv_c*=dc_x
   xv_c[1::2, :] += dc_x/2
   xv_c+=sim_params["R"]['c']
   xv_l, yv_l = np.meshgrid(np.arange(N_X_l), np.arange(N_Y_l), sparse=False, indexing='xy')
   yv_l*= dl_y
   yv_l+=sim_params["R"]['l']
   xv_l*=dl_x
   xv_l[1::2, :] += dl_x/2
   xv_l+=sim_params["R"]['l']
   yv_l+=np.max(yv_c)+sim_params["R"]['c']
   return xv_c.flatten(),yv_c.flatten(),xv_l.flatten(),yv_l.flatten()

def fig_split_sync():
   sim_params = json.load(open("/home/kodda/Dropbox/p2pflab/lettuces_and_cabbages/2D_geom/default.json"))
   L = 20
   W = L
   t_h = sim_params["tmax"]["c"]
   xs_c, ys_c, xs_l, ys_l = get_xy_split_sync(L, W, sim_params)

   plants = []
   for i in range(len(xs_c)):
      plants.append({'x': xs_c[i], 'y' : ys_c[i], 't': 0, 'species': 'c', "id": i})
   for i in range(len(xs_l)):
      plants.append({'x': xs_l[i], 'y' : ys_l[i], 't': 0, 'species': 'l', "id": i})

   plot_field(29.5, plants, sim_params, "split_sync.png", "N/t=%.2f"%((len(xs_c)+len(xs_l))/t_h))

def get_Nx_Ny_split_desync(L, W, sim_params):
   Rml = sim_params["R"]["l"]
   al = sim_params["a"]["l"] 
   Rmc = sim_params["R"]["c"]
   ac = sim_params["a"]["c"] 

   Rc = growth(30, 0, Rmc, ac)
   Rc_12 = growth(15, 0, Rmc, ac)
   Rl = growth(30, 0, Rml, al)
   Rl_12 = growth(15, 0, Rml, al)

   dc = (Rc+Rc_12)/np.sqrt(2)
   dl = (Rl+Rl_12)/np.sqrt(2)

   dc_x = 2*dc
   dc_y = dc
   dl_x = 2*dl
   dl_y = dl

   N_Y_l = L/(dc_x*dc_y/dl_x+dl_y)
   N_Y_c = N_Y_l*dc_x/dl_x

   N_X_c = L/dc_x
   N_X_l = L/dl_x

   print(np.floor(N_X_c)*N_Y_c, np.floor(N_X_l)*N_Y_l)

   return np.floor(N_X_c), np.floor(N_X_l), np.floor(N_Y_c)-1, N_Y_l, dc_x, dc_y, dl_x, dl_y

def get_xy_split_desync(L, W, sim_params):
   N_X_c, N_X_l, N_Y_c, N_Y_l, dc_x, dc_y, dl_x, dl_y = get_Nx_Ny_split_desync(L, W, sim_params)
   print("Nc", N_X_c*N_Y_c,"Nc", N_X_c*N_Y_c)

   xv_c, yv_c = np.meshgrid(np.arange(N_X_c), np.arange(N_Y_c), sparse=False, indexing='xy')
   ts_c = 15*(yv_c%2)
   ts_c = 15-ts_c

   yv_c*= dc_y
   yv_c+=sim_params["R"]['c']
   xv_c*=dc_x
   xv_c[1::2, :] += dc_x/2
   xv_c+=sim_params["R"]['c']
   xv_l, yv_l = np.meshgrid(np.arange(N_X_l), np.arange(N_Y_l), sparse=False, indexing='xy')
   ts_l = 15*(yv_l%2)
   ts_l = 15-ts_l
   yv_l*= dl_y
   yv_l+=sim_params["R"]['l']
   xv_l*=dl_x
   xv_l[1::2, :] += dl_x/2
   xv_l+=sim_params["R"]['l']
   yv_l+=np.max(yv_c)+sim_params["R"]['c']
   return xv_c.flatten(),yv_c.flatten(),xv_l.flatten(),yv_l.flatten(), ts_c.flatten(), ts_l.flatten() 

def fig_split_desync():
   sim_params = json.load(open("/home/kodda/Dropbox/p2pflab/lettuces_and_cabbages/2D_geom/default.json"))
   L = 20
   W = L
   t_h = sim_params["tmax"]["c"]
   xs_c, ys_c, xs_l, ys_l, ts_c, ts_l = get_xy_split_desync(L, W, sim_params)

   plants = []
   for i in range(len(xs_c)):
      plants.append({'x': xs_c[i], 'y' : ys_c[i], 't': ts_c[i], 'species': 'c', "id": i})
   for i in range(len(xs_l)):
      plants.append({'x': xs_l[i], 'y' : ys_l[i], 't': ts_l[i], 'species': 'l', "id": i})

   plot_field(29.5, plants, sim_params, "split_desync.png", "N/t=%.2f"%((len(xs_c)+len(xs_l))/t_h))

fig_mono_desync('c')
fig_mono_desync('l')
fig_mono_sync('c')
fig_mono_sync('l')
fig_mix_sync()
fig_mix_desync()
fig_split_sync()
fig_split_desync()

"""
K=50
Rms=np.linspace(.1, 1, K)
a=np.linspace(.1, 1, K)

Ns=np.zeros([K,K])
L=20
W=20

species = 'c'
sim_params = json.load(open("/home/kodda/Dropbox/p2pflab/lettuces_and_cabbages/2D_geom/default.json"))

for i in range(K):
   for j in range(K):
      sim_params["R"]["c"]=Rms[i]
      sim_params["a"]["c"]=Rms[i]
      
      Nx, Ny, R = get_Nx_Ny_mono_sync(L, W, species, sim_params)
      Nsync=Nx*Ny
      Nx, Ny, d, R = get_Nx_Ny_mono_desync(L, W, species, sim_params)
      Ndesync=Nx*Ny
      Ns[i,j]=Nsync-Ndesync
"""


"""
xv,yv=get_trigrid(L, W, R)

#xv2=xv+R
#yv2=yv+np.sqrt(3)/4

xs=xv.flatten()
ys=yv.flatten()#*.866*np.sqrt(3)/2
xs2=xv2.flatten()
ys2=yv2.flatten()

plants = []
for i in range(len(xs)):
   plants.append({'x': xs[i], 'y' : ys[i], 't': 0, 'species': "c", "id": i})

#for i in range(len(xs)):
#   plants.append({'x': xs2[i], 'y' : ys2[i], 't': 15, 'species': "l", "id": i})


mk_fig(29.5, plants, sim_params, "mono_c.png")
"""

"""
plants = []
species = 'l'
plants.append(Plant(0, 0, 0, 0, species, sim_params["R"][species], sim_params["a"][species], 30))

x=np.cos(np.pi/4)*(R+r)
y=np.sin(np.pi/4)*(R+r)

species='c'
plants.append(Plant(1, x, y, 0, species, sim_params["R"][species], sim_params["a"][species], 30))
plants.append(Plant(2, x, -y, 0, species, sim_params["R"][species], sim_params["a"][species], 30))
plants.append(Plant(3, -x, -y, 0, species, sim_params["R"][species], sim_params["a"][species], 30))
plants.append(Plant(4, -x, y, 0, species, sim_params["R"][species], sim_params["a"][species], 30))

pdicts=[p.to_dict() for p in plants]

mk_fig(29, pdicts, sim_params, "lala.png")

z =np.spqr((R+r)**2-R**2)

L = 10
W = 4
N_lines_c = 1

plants = []

Ncl = np.ceil(L/(2*R))

xs_c = R + 2*R*np.arange(Ncl)
xs_l = xs_c + r
ys_c = np.ones(Ncl)*R
ys_l = ys_c + z


cabbages=[]
lettuces=[]

#cabbages, lettuces = mk_line(cabbages,lettuces,x0,L,W,R,r)



L_fig=L+2*R
W_fig=N_lines*W+2*N_lines*R

fig=pl.figure(figsize=(L_fig,W_fig))
ax = fig.add_subplot(111)

fig.patch.set_facecolor("#805b21")
pl.axis("off")

pl.plot([-R,-R,L+R,L+R],[-R,N_lines*(W+2*R)-1,-R,N_lines*(W+2*R)-1],".")
#ax.set_aspect(1)

for p in cabbages:
   circle = pl.Circle(p, R, color=[0.1,.4,0.1])   
   ax.add_artist(circle)

for p in lettuces:
   circle = pl.Circle(p, r, color=[0.1,.8,0.1])   
   ax.add_artist(circle)

pl.savefig("lala.png", facecolor=fig.get_facecolor(),bbox_inches="tight")
"""