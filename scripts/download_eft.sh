#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.

set -ex

echo "Downloading EFT fitting data"
mkdir eft_fit
cd eft_fit

#COCO2014-All ver0.1
wget https://dl.fbaipublicfiles.com/eft/eft_fit_ver01/COCO2014-All-ver01.json

#COCO2014-Part ver0.1
wget https://dl.fbaipublicfiles.com/eft/eft_fit_ver01/COCO2014-Part-ver01.json

wget https://dl.fbaipublicfiles.com/eft/COCO2014-Val-ver10.json



# LSPet ver0.1 (outdated)
# wget https://dl.fbaipublicfiles.com/eft/eft_fit_ver01/LSPet_ver01.json

#MPII ver0.1
wget https://dl.fbaipublicfiles.com/eft/eft_fit_ver01/MPII_ver01.json

#PoseTrack ver0.1
wget https://dl.fbaipublicfiles.com/eft/eft_fit_ver01/PoseTrack_ver01.json

# LSPet ver1.0 (manual filtering)
wget https://dl.fbaipublicfiles.com/eft/LSPet_test_ver10.json
wget https://dl.fbaipublicfiles.com/eft/LSPet_train_ver10.json

# OCHuman ver1.0 (manual filtering)
wget https://dl.fbaipublicfiles.com/eft/OCHuman_test_ver10.json
wget https://dl.fbaipublicfiles.com/eft/OCHuman_train_ver10.json

echo "Done"
