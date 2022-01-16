# Running EFT in you local machines

Here, we show EFT fitting procedure by using COCO 2014 as example. 

You may use other dataset similarly.

We call the root folder of this repo as "."

## Additional Setting
- Download extra data from [SPIN](https://github.com/nkolot/SPIN):
```
sh scripts/download_spin_data.sh
```
You can see new files in "./extradata/spin/"

- Download preprocessed pose_regressor models
```
sh scripts/download_model_zoo.sh
```

- Set coco annotation and image files under eft_root/data/coco
```
.
├── ...
├── data_sets 
│   └── coco          
│        ├── annotations         # the folder with .json files. E.g., person_keypoints_train2014.json
│        └── train2014           # the folder with image files
└── ...
```
Note that you may use symbolic links to setup this folder structure


## Run preprocessing
- This process convert the raw 2D annotation data into a unified format, saved in npz
- Run the following 
``` 
python -m eft.db_processing.coco
```
- The output is saved in "./preprocessed_db/coco_2014_train_12kp.npz"


## Run EFT Fitting
- Run the following 
``` 
python -m demo.eftFitting
```
- You will see a GUI window to see EFT process. 

- Use mouse control to see the 3D in other views (see below about the key information)
- Toggle "C" in 3D Visualizer window to visualize 3D in image cooridnate
- Press "q" in 3D Visualizer window to move to the next example
 - Other key information:
   - mouse left + move: view change
   - mouse right + move: zoom in/out
   - shift + mouse left + move: pan
   - 'C': toggle between 3D view and image view
   - 'q': go to the next sample
   - 'w': toggle between solid mesh and wire-frame mesh
   - 'j': on/off for 3d skeleton
   - 'm': on/off for 3d mesh  
   - 'f': on/off for floor


## EFT outputs
- The EFT output is saved in "./eft_out" as PKL format.
- You can disable "--bDebug_visEFT" in eftFitting.py if you only want to get EFT outputs without visualization.

## License
[CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode). 
See the [LICENSE](LICENSE) file. 