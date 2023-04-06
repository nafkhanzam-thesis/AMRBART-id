#! /bin/bash

git clone https://github.com/nafkhanzam-thesis/AMRBART-v3
cd AMRBART-v3/wikification

pip install -r requirements.txt

git clone https://github.com/facebookresearch/BLINK.git
pushd BLINK
  chmod +x download_blink_models.sh
  ./download_blink_models.sh
popd

mkdir data
