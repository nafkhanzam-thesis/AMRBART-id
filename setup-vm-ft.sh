#! /bin/bash

git clone https://github.com/nafkhanzam-thesis/AMRBART-v3
cd AMRBART-v3

mkdir models
pushd models
  #~ Pre-trained model (mbart-large-50)
  # wget https://storage.nafkhanzam.com/thesis/backups/mbart-large-50-pretrained.tar.gz
  # tar -xvzf mbart-large-50-pretrained.tar.gz

  #~ Pre-trained model (mbart-en-id-smaller)
  # wget https://storage.nafkhanzam.com/thesis/backups/mbart-en-id-smaller-pre-trained.tar.gz
  # tar -xvzf mbart-en-id-smaller-pre-trained.tar.gz

  #~ Fine-tuned model (mbart-en-id-smaller after 1 epoch)
  epoch=16
  wget https://storage.nafkhanzam.com/thesis/backups/mbart-large-50-finetuned-e$epoch.tar.gz
  tar -xvzf mbart-large-50-finetuned-e$epoch.tar.gz

  #~ Fine-tuned model (mbart-en-id-smaller after 1 epoch)
  # epoch=1
  # wget https://storage.nafkhanzam.com/thesis/backups/mbart-en-id-smaller-pre-trained-fine-tune-e$epoch.tar.gz
  # tar -xvzf mbart-en-id-smaller-pre-trained-fine-tune-e$epoch.tar.gz
popd

mkdir datasets
pushd datasets
  wget https://storage.nafkhanzam.com/thesis/backups/amrbart-new.tar.gz
  tar -xvzf amrbart-new.tar.gz
popd

pip install -r requirements.txt

# ./train.sh mbart-large-50-pretrained
# ./eval.sh mbart-large-50-finetuned-e16
