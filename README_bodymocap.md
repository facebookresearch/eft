# Motion Capture Demo

We have released pre-trained 3D pose estimation models trained with our EFT dataset. Our pose estimation code is based on [SPIN](https://github.com/nkolot/SPIN) with modifications

<p>
    <img src="https://github.com/jhugestar/jhugestar.github.io/blob/master/img/eft_bodymocap.gif" height="256">
</p>

## Requirements
Run the following script
```
sh scripts/download_mocapdata.sh
```
- This script downloads:
    - Extra data from [SPIN](https://github.com/nkolot/SPIN) at ./extradata/data_from spin
        - "J_regressor_extra.npy" and "smpl_mean_params.npz"
    - Pretrained model at ./models_eft
    - Sample videos at ./sampledata

## Installing third-party tools for bbox detection
- Our 3D body mocap demo assumes a bounding box. For this, you need either of the following options.
- (Option 1) by using [2D keypoint detector](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch): 
    - Run the following script
    ```
    sh scripts/install_pose2d.sh
    ```
    
- (Option 2) by using [YOLO](https://github.com/eriklindernoren/PyTorch-YOLOv3)
   - git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
   - Run the following script
    ```
    sh scripts/install_yolo.sh
    ```
   - Open demo/demo_bodymocap.py and modify BodyBboxDetector('2dpose') to BodyBboxDetector('yolo')
   ```
   #bboxdetector =  BodyBboxDetector('2dpose')      #original code
   bboxdetector =  BodyBboxDetector('yolo')      #new code
   ```

- (Option 3) Use your own bbox detection, export it, and load from the exported bbox file (TODO)

## A Quick Start
- Run the following. The mocap output will be shown on your screen
```
    # The output images are also saved in ./mocap_output
    python -m demo.demo_bodymocap --vPath ./sampledata/han_short.mp4 --renderout ./mocap_output

    #If you do not spefify the output folder, you can use a GUI mode
    python -m demo.demo_bodymocap --vPath ./sampledata/han_short.mp4 

    #Speficy the pretrained-model path, if necessary
    python -m demo.demo_bodymocap --vPath ./sampledata/han_short.mp4 --renderout ./mocap_output --checkpoint models_eft/2020_05_31-00_50_43-best-51.749683916568756.pt
```

- If you do not have a screen, use "xvfb-run" tool
```
    # The output images are also saved in ./mocap_output
    xvfb-run -a python -m demo.demo_bodymocap --vPath ./sampledata/han_short.mp4 --renderout ./mocap_output
```
## Run demo with a webcam
- Run,
```
    python -m demo.demo_bodymocap --webcam
```
- Press "C" in the 3D window to see the scene in camera view (See below for GUI key information)


## GUI mode 
- In GUI mode, you can use mouse and keyboard to change view point. 
    - This mode requires a screen connected to your machine 
    - Keys in OpenGL 3D window
        - mouse-Left: view rotation
        - mouse-Right: view zoom chnages
        - shift+ mouseLeft: view pan
        - C: toggle for image view/3D free view
        - w: toggle wireframe/solid mesh
        - j: toggle skeleton visualization
        - R: automatically rotate views
        - f: toggle floordrawing
        - q: exit program


## Other options 
- `--webcam`: Run demo for a video file  (without using `--vPath` option)
- `--vPath /your/path/video.mp4`: Run demo for a video file
- `--vPath /your/dirPath`: Run demo for a folder that contains image seqeunces
- `--download --url https://videourl/XXXX`: download public videos via `youtube-dl` and run with the downloaded video. (need to install youtube-dl first)

- `--renderout ./outputdir`: Save the output images into files
- `--pklout ./outputdir`: Save the pose reconstruction data (SMPL parameters and vertices) into pkl files
- `--startFrame 100 --endFrame 200`: Specify start and end frames (e.g., 100th frame and 200th frame in this example)
- `--single`: To enforce single person mocap (to avoid outlier bboxes). This mode chooses the biggest bbox. 



## Mocap output format (pkl)
As output, the 3D pose estimation data per frame is saved as a pkl file. Each person's pose data is saved as follows:
```
mocap_single = {
        'pred_rotmat': predoutput['pred_rotmat'],           #(1, 24,3, 3)
        'pred_betas': predoutput['pred_betas'],             #(1,10)
        'pred_camera': predoutput['pred_camera'],           #[cam_scale, cam_offset_x,, cam_offset_y ]
        'pred_vertices_imgspace': predoutput['pred_vertices_img'],  #3D SMPL vertices where X,Y are aligned to images
        'pred_joints_imgspace': predoutput['pred_joints_img'],      #3D joints where X,Y are aligned to images
        'bbox_xyxy': predoutput['bbox_xyxy'],        #[minX,minY,maxX,maxY]

        'bboxTopLeft': predoutput['bboxTopLeft'],   #(2,)       #auxiliary data used inside visualization
        'boxScale_o2n': predoutput['boxScale_o2n']      #scalar #auxiliary data used inside visualization          
}
```
## Load saved mocap data (pkl file)
- Run the following code to load and visualize saved mocap data files
```
#./mocap_output/mocap is the directory where pkl files exist
python -m  demo.demo_loadmocap --mocapdir ./mocap_output/mocap
```
- Note: current version use GUI mode for the visualization (requiring a screen). 
- TODO: screenless rendereing. Saved as images and videos

## Run demo with SMPL-X model (TODO)
- Current code is based on SMPL model, but you can run with SMPL-X model
- Make sure to use the pretrained network model with SMPL-X model (TODO: will be available)
- Run,
```
    python -m demo.demo_mocap --vPath /your/path/video.mp4 --bUseSMPLX
```