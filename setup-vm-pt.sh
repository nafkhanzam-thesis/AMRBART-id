#! /bin/bash

git clone https://github.com/nafkhanzam-thesis/AMRBART-v3
cd AMRBART-v3

sudo apt install git-lfs
git lfs install

mkdir models
pushd models
  #~ mBART-large-50
  git clone https://huggingface.co/facebook/mbart-large-50

  #~ mBART-en-id
  # wget https://storage.nafkhanzam.com/thesis/backups/mbart-en-id-smaller.tar.gz
  # tar -xvzf mbart-en-id-smaller.tar.gz
popd

mkdir datasets
pushd datasets
  wget https://storage.nafkhanzam.com/thesis/backups/amrbart-new.tar.gz
  tar -xvzf amrbart-new.tar.gz
popd

pip install -r requirements.txt
