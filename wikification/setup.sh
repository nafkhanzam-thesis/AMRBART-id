#! /bin/bash

git clone https://github.com/nafkhanzam-thesis/AMRBART-v3
cd AMRBART-v3/wikification
mkdir data

pip install -r requirements.txt

git clone https://github.com/facebookresearch/BLINK.git
pushd BLINK
  chmod +x download_blink_models.sh
  ./download_blink_models.sh
popd

#~ send
# scp ~/kode/nafkhanzam/thesis/model-sources/model-amrbart-amr3-aug/val_outputs/test_generated_predictions_142400.txt user@216.153.52.232:/home/user/AMRBART-v3/wikification/data/input.amr

#~ receive
# scp user@216.153.52.232:/home/user/AMRBART-v3/wikification/data/input.amr.wiki ~/kode/nafkhanzam/thesis/model-sources/model-amrbart-amr3-aug/val_outputs/test.amr.wiki
