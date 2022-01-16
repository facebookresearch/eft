import os
from tqdm import tqdm


coco2014= '/run/media/hjoo/disk/data/coco/val2014'
coco2017= '/run/media/hjoo/disk/data/coco2017/val2017'
coco2017_list = os.listdir(coco2017)

for name in tqdm(coco2017_list):
    # newName = os.path.join(coco2014,os.path.basename(name)[15:])

    newName = os.path.join(coco2014,"COCO_val2014_"+os.path.basename(name))

    if os.path.exists(newName) ==False:
        print(f"Img doesn't exist: {newName}")
        assert False
    else:
        print(f"Img exist!: {newName}")

