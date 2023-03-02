#! /bin/bash

git clone https://github.com/nafkhanzam-thesis/AMRBART-v3
cd AMRBART-v3

mkdir models
pushd models
  #~ Pre-trained model
  # wget https://storage.nafkhanzam.com/thesis/backups/mbart-en-id-smaller-pre-trained.tar.gz
  # tar -xvzf mbart-en-id-smaller-pre-trained.tar.gz

  #~ Fine-tuned model
  epoch=1
  wget https://storage.nafkhanzam.com/thesis/backups/mbart-en-id-smaller-pre-trained-fine-tune-e$epoch.tar.gz
  tar -xvzf mbart-en-id-smaller-pre-trained-fine-tune-e$epoch.tar.gz
popd

mkdir datasets
pushd datasets
  wget https://storage.nafkhanzam.com/thesis/backups/amrbart-new.tar.gz
  tar -xvzf amrbart-new.tar.gz
popd

pip install -r requirements.txt

# ./eval.sh mbart-en-id-smaller-pre-trained-fine-tune
