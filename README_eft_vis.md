# Getting Started with EFT Dataset

## Visualizing EFT Fitting Results
- Run "demo/visEFTFit.py" 
- set EFT fitting dir path and image path. 
```
python -m demo.visEFTFit --rendermode geo --img_dir (your_img_dir) --fit_data (downloaded_fitting_location)
#for example
python -m demo.visEFTFit --rendermode geo --img_dir ~/data/coco/train2014 --fit_data eft_fit/COCO2014-Part-ver01.json
```

- The rendered ouptut is written in default location "render_eft". You can change it with "--render_dir (outputFolderName)"

- More examples with other options:

```
#Render via Phong shading, on original image space
python -m demo.visEFTFit --rendermode geo

#Render via denspose IUV, on bbox image
python -m demo.visEFTFit --rendermode denspose --onbbox

#Use --waitforkeys to stop after visualizing each sample, waiting for any key pressed
python -m demo.visEFTFit --rendermode geo --waitforkeys

#use --turntable to show the mesh via turn table views
python -m demo.visEFTFit --rendermode geo --turntable

#Use --onbbox, if you want to visualize output in bbox space 
python -m demo.visEFTFit --onbbox

#Use --multi to show all annotated humans (multiple people) per image
python -m demo.visEFTFit --rendermode normal --multi

#Use --multi --turntable
python -m demo.visEFTFit --rendermode geo --multi --turntable

```
- By default, the rendered images are saved in the directory (default: ./render_eft ) specified by "--render_dir"

- Our visualization pipeline uses OpenGL, requiring a screen to dispaly output. For screenless rendering (e.g., a server without a screen), use "xvfb-run"
```
xvfb-run python -m demo.visEFTFit
```

- In the EFT files for COCO, you can find coco annotation ID in 'annotId'. 
Check the following example script to visualize EFT fitting with other COCO annotations (e.g., bbox, skeleton)
```
python -m demo.visEFTFit_coco --cocoAnnotFile (your_coco_path)/annotations/person_keypoints_train2014.json

```

## Visualizing EFT Fitting with GUI mode
<p>
    <img src="https://github.com/jhugestar/jhugestar.github.io/blob/master/img/eft_gui_viewer_2.gif" height="256">
</p>

- In this visualization, you can use your keyboard and mouse to see the fitting output from any view point.
- Script: "demo/visEFTFit_gui.py" 

- Run,
```
 python -m demo.visEFTFit_gui --img_dir /your/dataset/path --fit_dir /your/dataset/path 

 #For example
 python -m demo.visEFTFit_gui --img_dir ~/data/coco/train2014 --fit_dir ~/CVPR2020_submit_fits/11-08_coco_with8143 
```
- The default visualization will show a single person at each time
- You can use "--multi" option to visualize all (reconstructed) humans for an image
```
  python -m demo.visEFTFit_gui --multi --img_dir ~/data/coco/train2014 --fit_dir ~/CVPR2020_submit_fits/11-08_coco_with8143 
```

 - If you can see a 3D view, good! You can use mouse and keyboard to change viewpoint
 - In the 3d view, press 'q' to go to the next sample
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


## License
[CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode). 
See the [LICENSE](LICENSE) file. 