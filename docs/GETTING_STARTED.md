# Getting Started with EFT Dataset

## Visualize EFT Fitting Results
- Run "demo/visEFTFit.py" 
- set EFT fitting dir path and image path. 
```
python -m demo.visEFTFit --rendermode geo --img_dir (your_img_dir) --fit_dir (downloaded_fitting_location)
```
- The rendered ouptut is written in default location "render_eft". You can change it with "--render_dir (outputFolderName)"

- More examples with other options:

```
#Render via Phong shading, on original image space
python -m demo.visEFTFit --rendermode geo

#Render via denspose IUV, on bbox image
python -m demo.visEFTFit --rendermode denspose --onbbox

#Render via normal shading, all annotated humans per image
python -m demo.visEFTFit --rendermode normal --bShowMultiSub
```
- When windows pop up, press any key to move on.
- For screenless rendering (e.g., a server without a screen), use "xvfb-run"
```
xvfb-run python -m demo.visEFTFit
```
- In the EFT files for COCO, you can find coco annotation ID in 'annotId'. 
Check the following script to visualize EFT fitting with other COCO annotations (e.g., bbox, skeleton, or mask)
```
python -m demo.visEFTFit_coco --cocoAnnotFile (your_coco_path)/annotations/person_keypoints_train2014.json

```

## Visualize EFT Fitting with GUI mode
- Script: "demo/visEFTFit_gui.py" 

- Run,
```
 python -m demo.visEFTFit_gui --img_dir /your/dataset/path --fit_dir /your/dataset/path --smpl_dir /your/dataset/path
 #For example
 python -m demo.visEFTFit_gui --img_dir ~/data/coco/train2014 --fit_dir ~/CVPR2020_submit_fits/11-08_coco_with8143 --smpl_dir ./smpl
```
- The default visualization will show all recontructed humans in each image. 
- You can use "--bShowSingle" option to visualize a single human at each time
```
  python -m demo.visEFTFit_gui --img_dir ~/data/coco/train2014 --fit_dir ~/CVPR2020_submit_fits/11-08_coco_with8143 --smpl_dir ./smpl --bShowSingle
```

 - If you can see 3D view, good! You can use mouse and keyboard to change viewpoint (see https://github.com/fairinternal/glViewer for the key info.)
 - In the 3d view, press 'q' to go to the next sample
 - Some other key info
   - 'C': toggle between 3D view and image view
   - 'q': go to the next sample
   - 'w': toggle between solid mesh and wire-frame mesh
   - 'j': on/off for 3d skeleton
   - 'm': on/off for 3d mesh  
