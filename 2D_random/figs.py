import utils as ut
import json

data=json.load(open("/home/kodda/Dropbox/p2pflab/lettuces_and_cabbages/2D_random/res/20x20_sel.json"))

ts = [0,200]
ut.mk_anim(data["plants"], data["sim_params"], ts, folder = "tmp", svg="test.mp4")