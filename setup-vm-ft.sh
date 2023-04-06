#! /bin/bash

git clone https://github.com/nafkhanzam-thesis/AMRBART-v3
cd AMRBART-v3

mkdir models
pushd models
  #~ Pre-trained model (mbart-large-50)
  # wget https://storage.nafkhanzam.com/thesis/models/mbart-large-50-pretrained.tar.gz
  # tar -xvzf mbart-large-50-pretrained.tar.gz

  #~ Pre-trained model (mbart-en-id-smaller)
  # wget https://storage.nafkhanzam.com/thesis/models/mbart-en-id-smaller-pretrained.tar.gz
  # tar -xvzf mbart-en-id-smaller-pretrained.tar.gz

  #~ Fine-tuned model (mbart-en-id-smaller after 16 epoch)
  wget https://storage.nafkhanzam.com/thesis/models/mbart-large-50-finetuned.tar.gz
  tar -xvzf mbart-large-50-finetuned.tar.gz

  #~ Pre-trained concat model (mbart-en-id-smaller)
  # wget https://storage.nafkhanzam.com/thesis/models/mbart-en-id-smaller-concat-pretrained.tar.gz
  # tar -xvzf mbart-en-id-smaller-concat-pretrained.tar.gz
popd

mkdir datasets
pushd datasets
  #~ AMRBART
  wget https://storage.nafkhanzam.com/thesis/backups/amrbart-new.tar.gz
  tar -xvzf amrbart-new.tar.gz

  #~ Concat
  # wget https://storage.nafkhanzam.com/thesis/backups/amrbart-concat.tar.gz
  # tar -xvzf amrbart-concat.tar.gz
popd

pip install -r requirements.txt

# ~ AMRBART
# ./train.sh mbart-en-id-smaller-pretrained
# ./eval.sh mbart-large-50-finetuned

# ~ Concat
# IS_CONCAT=1 ./train.sh mbart-en-id-smaller-concat-pretrained amrbart-concat
