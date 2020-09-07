# 3D Human Pose Data Format

EFT Fitting data contains [SMPL](https://smpl.is.tue.mpg.de/) model parameters corresponding to each human instance in the public 2D keypoint dataset. 
Note that these are not ground-truth! However, we found that the quality is good enough to train a strong 3D human pose regressor. 

## EFT Fitting data format (json)
Each json file contains a list of 3D human pose parameters. 
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
    #data['smpltype']           #smpl or smplx
    #data['annotId']            #Only for COCO dataset. COCO annotation ID
    #data['imageName']          #image name (basename only)

    #data['subjectId']          #(optional) a unique id per sequence. usually {seqName}_{id}

    pass
```

## Skeleton Format
SMPL model generates 3D mesh structure, and the order of 3D joints obtained from the mesh can vary based on the choice of regressor matrix and application. 

The original SMPL model has 45 joints. In our example, we use [OpenPose18](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md) (similar to COCO)joint ordering. See [jointorders.py](https://github.com/facebookresearch/eft/blob/master/eft/cores/jointorders.py). 


## Bodymocap output format 
- You can export [bodymocap](https://github.com/facebookresearch/eft/blob/bodymocap/README_bodymocap.md) output as pkl files (with --pklout flag)
- The basic formtion is the same as json, but each pkl file contains 3D poses per each image. See [bodymocap](https://github.com/facebookresearch/eft/blob/master/README_bodymocap.md#load-saved-mocap-data-pkl-file)


## BBox format (json)
- You can export/load bbox for body mocap
- Format (json):
```
{"imgPath": imagePath, "bboxes_xywh": list of bboxes_xywh}
```
- For example
```
{"imgPath": "/run/media/hjoo/disk/data/Penn_Action/frames/0001/000151.jpg", "bboxes_xywh": [[150, 126, 153, 184]]}
```
