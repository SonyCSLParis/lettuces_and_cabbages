import numpy as np

def growth(t, t0, species, R, a, static = False):
    if not(static):
        return R[species]/(1+np.exp(-a[species]*(t-t0)+4))
    else:
        return R[species]
    
def is_intersect(p1, p2, t, R, a, static = False):
    r1 = growth(t, p1.t, p1.species, R, a, static)
    r2 = growth(t, p2.t, p2.species, R, a, static)
    d=np.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2)
    return d<(r1+r2)

def get_meas(plants, t, R, a, static = False):
   Nc, Nl, Ac, Al = 0, 0, 0, 0
   for p in plants:
       r = growth(t, p.t, p.species, R, a, static)
       if p.species == "c":
         Nc += 1
         Ac += np.pi*r**2
       if p.species == "l":
         Nl += 1
         Al += np.pi*r**2
   return [Nc, Nl, Ac, Al]
