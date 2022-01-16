#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.

set -ex

echo "Downloading Pre-trained models"
mkdir eft_model_zoo
cd eft_model_zoo

wget https://dl.fbaipublicfiles.com/eft/eft_model_zoo/mpii.pt       # A poor model
wget https://dl.fbaipublicfiles.com/eft/eft_model_zoo/cocoall_h36m_mpiinf_posetrack_lsptrain_ochuman.pt     # A strong model

# You can also donwload the following models
# wget https://dl.fbaipublicfiles.com/eft/eft_model_zoo/h36m.pt
# wget https://dl.fbaipublicfiles.com/eft/eft_model_zoo/coco-all.pt
# wget https://dl.fbaipublicfiles.com/eft/eft_model_zoo/cocopart.pt
# wget https://dl.fbaipublicfiles.com/eft/eft_model_zoo/cocoall_h36m_mpiinf.pt


echo "Done"
