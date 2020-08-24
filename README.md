# [Exemplar Fine-Tuning for 3D Human Pose Fitting Towards In-the-Wild 3D Human Pose Estimation](https://arxiv.org/abs/2004.03686)

This repository contains pseudo-GT 3D human pose data produced by [Exemplar Fine-Tuning (EFT)](https://arxiv.org/abs/2004.03686) method for in-the-wild 2D images. The 3D pose data is in the form of [SMPL](https://smpl.is.tue.mpg.de/) parameters, and this can be used as a supervision to train a 3D pose estimation algiritm (e.g., [SPIN](https://github.com/nkolot/SPIN) or [HMR](https://github.com/akanazawa/hmr)). We found that our EFT dataset is sufficient to build a model that is comparable to the previous SOTA algorithms without using any other indoor 3D pose dataset. See our [paper](https://arxiv.org/abs/2004.03686) for more details.
<p>
    <img src="docs/example1.jpg" height="256">
    <img src="docs/example2.jpg" height="256">
    <img src="docs/3432.gif" height="256">
</p>

## Installing Requirements
It is convenient and safe to use conda environment
```
conda create -n venv_eft python=3.6
conda activate venv_eft
pip install -r requirements.txt
```

## Download EFT Fitting data (json formats)
This repository only provides corresponding SMPL parameters for public 2D keypoint datasets (such as [COCO](https://cocodataset.org/), [MPII](http://human-pose.mpi-inf.mpg.de/)). You need to download images from the original dataset website.

Run the following script to download our EFT fitting data:
```
sh scripts/download_eft.sh 
```
   - The EFT data will be saved in ./eft_fit/(DB_name).json. Each json file contains a version EFT fitting for a public dataset. 
   - See [Data Format](docs/README_dataformat.md) for details
   - Currently available EFT fitting outputs (cvpr submit version):


|Dataset Name   |  SampleNum | Version    | Manual Filtering |         File Name         |
|---------------| -----------| ---------  | ---------------- |-------------------------- |
|COCO2014-12kp  | 28K        | 0.1        | No               |  COCO2014-All-ver01.json  |
|COCO2014-6kp   | 79K        | 0.1        | No               |  COCO2014-Part-ver01.json |
|MPII           | 14K        | 0.1        | No               |  MPII_ver01.json          |
|LSPet          | 7K         | 0.1        | No               |  LSPet_ver01.json         |

  - COCO2014-All-ver01.json: [COCO](https://cocodataset.org/#home) 2014 training set by electing the samples 6 keypoints or more keypoints are annotated.
  - COCO2014-Part-ver01.json: [COCO](https://cocodataset.org/#home) 2014 training set by selecting the sample that 12 limb keypoints or more are annotated.
  - MPII_ver01.json : [MPII](http://human-pose.mpi-inf.mpg.de/) Keypoint Dataset
  - LSPet_ver01.json : [LSPet](https://sam.johnson.io/research/lspet.html) Dataset
  - [PanopticStudio DB](http://domedb.perception.cs.cmu.edu/): TBA
  - Note that the number of samples are fewer than the original sample numbers in each DB, since we automatically filtered out bad samples
  - Manual Filtering: we plan to filter out erroneous results by manual annotations 

### Download Other Required Data
- SMPL Model (Neutral model: basicModel_neutral_lbs_10_207_0_v1.0.0.pkl):
    - Download in the original [website](http://smplify.is.tue.mpg.de/login). You need to register to download the SMPL data.
    - Put the file in: ./extradata/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl

- Densepose (optional, for Densepose rendering): 
  - Run the following script
  ```
      sh scriptsdownload_dp_uv.sh    
  ```
  - Files are saved in ./extradata/densepose_uv_data/
  
## Download Images from Original Public DB website
 - [COCO](https://cocodataset.org/#home): [2014 Training set](http://images.cocodataset.org/zips/train2014.zip)
 - [MPII](http://human-pose.mpi-inf.mpg.de/): [Download Link](https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz)
 - [LSPet](https://sam.johnson.io/research/lspet.html): [Download Link](http://datasets.d2.mpi-inf.mpg.de/hr-lspet/hr-lspet.zip)

## Visualize EFT Fitting Results
- See [GETTING_STARTED](docs/GETTING_STARTED.md)

## Citation
```
@article{joo2020eft,
  title={Exemplar Fine-Tuning for 3D Human Pose Fitting Towards In-the-Wild 3D Human Pose Estimation},
  author={Joo, Hanbyul and Neverova, Natalia and Vedaldi, Andrea},
  journal={arXiv preprint arXiv:2004.03686},
  year={2020}
}
```

## License
[CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode). 
See the [LICENSE](LICENSE) file. 


## References
The body mocap code is a modified version of [SPIN](https://github.com/nkolot/SPIN), and the majority of this code is borrowed from it.