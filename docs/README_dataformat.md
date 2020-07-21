# EFT Data format

EFT Fitting data contains [SMPL](https://smpl.is.tue.mpg.de/) model parameters corresponding to each human instance in the public 2D keypoint dataset. 
Note that these are not ground-truth! However, we found that the quality is good enough to train a strong 3D human pose regressor. 

## Json Format
Each Json file contains a list of 3D human pose parameters. 
```
eft_data = json.load(f)
print("EFT data: ver {}".format(eft_data['ver']))
eft_data_all = eft_data['data']       

for idx, data in enumerate(eft_data_all):
    #data['parm_pose']     #24x3x3, 3D rotation matrix for 24 joints
    #data['parm_shape']       #10 dim vector
    #data['parm_cam']        #weak-perspective projection: [cam_scale, cam_transX, cam_tranY]
    #data['bbox_scale']     #float
    #data['bbox_center']    #[x,y]
    #data['joint_validity_openpose18']      #Joint validity in openpose18 joint order
    #data['smpltype']           #SMPL or SMPL-X
    #data['annotId']            #Only for COCO dataset. COCO annotation ID
    #data['imageName']          #image name (basename only)

    pass
```

## Skeleton Format
SMPL model generates 3D mesh structure, and the order of 3D joints obtained from the mesh can vary based on the choice of regressor matrix and application. 

The original SMPL model has 45 joints. In our example, we use [OpenPose18](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md) (similar to COCO)joint ordering. See [jointorders.py](https://github.com/facebookresearch/eft/blob/master/eft/cores/jointorders.py). 
