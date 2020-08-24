# Body Pose Estimation

## Required
'./data/J_regressor_extra.npy'
'./data/smpl_mean_params.npz'

via wget http://visiondata.cis.upenn.edu/spin/data.tar.gz && tar -xvf data.tar.gz && rm data.tar.gz

## Testing

# Run Demos

## Additional Dependency
- We use 2D keypoint detector as bbox detector: 
    - git clone https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
    - (Internal only): You need download pretrained model and additional files: (Link)[https://www.dropbox.com/s/commi4ej1byewcf/light-pose-data.zip?dl=0]
            - Put this on "lightweight-human-pose-estimation" folder
            - lightweight-human-pose-estimation/demo_image.py
            - lightweight-human-pose-estimation/body2d_models
            - lightweight-human-pose-estimation/pretrain
            - Open "lightweight-human-pose-estimation.pytorch/demo_image.py" and set "checkpoint=" to your pretrain data
    - Open "demo/demo_mocap.py", and
        - Set the "pose2d_estimator_path=/your/path/lightweight-human-pose-estimation"
    
       
- Alternatively, you can use YOLO (https://github.com/eriklindernoren/PyTorch-YOLOv3)
   - This is for human bbox detection
   - git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
   - In yolo directory, download weight files
        - cd weights; bash download_weights.sh
   - Open "demo/demo_mocap.py", and
        - Set the "yolo_path=/your/path/PyTorch-YOLOv3"

## Run demo for a video file
- Run,
```
     python -m demo.demo_mocap --vPath /your/path/video.mp4 --checkpoint','/yourpath/checkpoints/2019_11_14-02_14_28-best-55.79321086406708.pt
```

- When you see a window popped up, type any keyboard key to start


## GUI mode vs screenless mode
- The default mode is a GUI mode where you can use mouse and keyboard to change view point. 
    - This mode required a screen (opengl requirement)
    - GUI for 3D window
        - mouseLeft: view rotation
        - mouseRight: view zoom chnages
        - shift+ mouseLeft: view pan
        - C: toggle for image view/3D free view
        - w: toggle wireframe/solid mesh
        - j: toggle skeleton visualization
        - R: automatically rotate views
        - f: toggle floordrawing
        - q: exit program

- You set a folder where you want to save the rendered images, and this mode can be run without a screen
```
    python -m demo.demo_mocap --vPath /your/path/video.mp4  --out ./outputFolder

    # or use xvfb-run  if you do not have connected screen to the machine(e.g., a remote server)
    xvfb-run python -m demo.demo_mocap --vPath /your/path/video.mp4  --out ./outputFolder
```
    - The rendered output is saved in ./outputFolder

# Other Options

## Run with Devfair Cluster
- See "run_ava.sh"

## Run demo for image frames in a folder
- Run,
    ```
     python -m demo.demo_mocap --vPath /your/path/images_%08d.jpg

     #Or
     python -m demo.demo_mocap --vPath /your/dirPath    

     #Specify start and end frame
     python -m demo.demo_mocap --vPath /your/dirPath    --startFrame 100 --endFrame 200
    ```

- When you see a window popped up, type any keyboard key to start



## Run demo with video urls (with video download via youtube-dl)
- Run,
    ```
     python -m demo.demo_mocap --download --url https://www.youtube.com/watch?v=c5nhWy7Zoxg
     ```

- When you see a window popped up, type any keyboard key to start

- This command will automatically download the video from the url in "./webvideos" folder. 


## Run demo with video urls (no video download)
- Run,
    ```
     python -m demo.demo_mocap --url https://www.youtube.com/watch?v=c5nhWy7Zoxg
     ```

- When you see a window popped up, type any keyboard key to start

- (Warning) This command may not work due to maybe opencv version issue. Then, try it with "--download" option, as described above


## Run demo with a webcam
- Run,
    ```
    python -m demo.demo_mocap --webcam
    ```

- When you see a window popped up, type any keyboard key to start

## Export mocap output as video
- Use option --out
```
python -m demo.demo_mocap --vPath /your/path/video.mp4  --out {output_folder}
```

## Export mocap output as pkl
- Use both options --out and --export


## Run demo by using SMPL-X 
- Run,
    ```
     python -m demo.demo_mocap --vPath /your/path/video.mp4 --bUseSMPLX
     ```

- Note that you use a model trained with SMPL-X: https://www.dropbox.com/s/nfsee5hmhtgy39l/03-28-46060-w_spin_mlc3d_46582-2089.zip?dl=0


## Run hand + body demo
- Note that you use a model trained with SMPL-X: https://www.dropbox.com/s/nfsee5hmhtgy39l/03-28-46060-w_spin_mlc3d_46582-2089.zip?dl=0
- You also need pose detector: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
- And handmocap: https://github.com/fairinternal/handmocap
- Note: the code is not complete. you need to change the path inside the demo_mocap_hand.py code
- Run,
```
    #webcam input
    python -m demo.demo_mocap_hand  --checkpoint ./spinmodel_shared/smplx-03-28-46060-w_spin_mlc3d_46582-2089/checkpoints/2020_03_28-21_56_16.pt 
                                    --bUseSMPLX --webcam

    #webcam show hand only
    python -m demo.demo_mocap_hand  --checkpoint ./spinmodel_shared/smplx-03-28-46060-w_spin_mlc3d_46582-2089/checkpoints/2020_03_28-21_56_16.pt 
                                    --bUseSMPLX --webcam --showHandonly

    #Input images in a directory  examples
    python -m demo.demo_mocap_hand  --checkpoint ./spinmodel_shared/smplx-03-28-46060-w_spin_mlc3d_46582-2089/checkpoints/2020_03_28-21_56_16.pt 
                                    --bUseSMPLX --vPath /target/dir/withImages
```                                    

