import matplotlib.pyplot as pl
import utils
import imageio
import glob

# permet de creer une image à l'instant t de la surface avecc les deux types
# de plantes

def mk_fig(t, plants, svg, R, a, static, cols):
   fig=pl.figure(figsize=(10,4))
   ax = fig.add_subplot(111)

   fig.patch.set_facecolor("#805b21")
   pl.axis("off")

   pl.plot([0,0,10,10],[-2,2,-2,2],".")
   #ax.set_aspect(1)

   for p in plants:
      r = utils.growth(t, p.t, p.species, R, a, static)
      circle = pl.Circle([p.x, p.y], r, color=cols[p.species])
      ax.add_artist(circle)
   pl.savefig(svg, facecolor=fig.get_facecolor(),bbox_inches="tight")
   pl.clf()

#collecte tous les fichiers images et  genere un film

def mk_anim(folder, svg):   
   files = glob.glob(folder+"/*")
   files.sort()
   image_list = [imageio.imread(f) for f in files]
   imageio.mimwrite(svg, image_list)
