#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.

set -ex

echo "Downloading SPIN data fitting data (https://github.com/nkolot/SPIN)"
cd extradata
wget http://visiondata.cis.upenn.edu/spin/data.tar.gz && tar -xvf data.tar.gz && rm data.tar.gz
mv -v data spin

echo "Done"
