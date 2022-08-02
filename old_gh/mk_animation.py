import imageio
import glob

folder = "/home/kodda/Dropbox/p2pflab/intercrop_comp/dynamic/tmp/"
files = glob.glob(folder+"/*")
files.sort()

image_list = [imageio.imread(f) for f in files]
imageio.mimwrite('rate_1.avi', image_list)

