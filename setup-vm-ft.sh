#! /bin/bash

git clone https://github.com/nafkhanzam-thesis/AMRBART-v3
cd AMRBART-v3

sudo apt install git-lfs
git lfs install

mkdir models
pushd models
  #~ Pre-trained model
  wget https://storage.nafkhanzam.com/thesis/backups/mbart-en-id-smaller-pre-trained.tar.gz
  tar -xvzf mbart-en-id-smaller-pre-trained.tar.gz

  #~ Fine-tuned model
  epoch=1
  wget https://storage.nafkhanzam.com/thesis/backups/mbart-en-id-smaller-pre-trained-fine-tune-e$epoch.tar.gz
  tar -xvzf mbart-en-id-smaller-pre-trained-fine-tune-e$epoch.tar.gz
popd

mkdir datasets
pushd datasets
  wget https://storage.nafkhanzam.com/thesis/backups/amrbart-splitted.tar.gz
  tar -xvzf amrbart-splitted.tar.gz
popd

pip install -r requirements.txt
