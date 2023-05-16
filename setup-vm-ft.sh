#! /bin/bash -i

# mkdir -p ~/miniconda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# rm -rf ~/miniconda3/miniconda.sh
# ~/miniconda3/bin/conda init bash
# source ~/.bashrc
# conda install pytorch=*=*cuda* cudatoolkit -c pytorch -y

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
  # wget https://storage.nafkhanzam.com/thesis/models/mbart-large-50-finetuned.tar.gz
  # tar -xvzf mbart-large-50-finetuned.tar.gz

  #~ Pre-trained concat model (mbart-en-id-smaller)
  # wget https://storage.nafkhanzam.com/thesis/models/mbart-en-id-smaller-concat-pretrained.tar.gz
  # tar -xvzf mbart-en-id-smaller-concat-pretrained.tar.gz

  # wget https://storage.nafkhanzam.com/thesis/models/model-amrbart-amr2.tar.gz
  # tar -xvzf model-amrbart-amr2.tar.gz
popd

mkdir datasets
pushd datasets
  wget https://storage.nafkhanzam.com/thesis/backups/amrbart-amr3-augfil.tar.gz
  tar -xvzf amrbart-amr3-augfil.tar.gz
popd

pip install -r requirements.txt

# ~ AMRBART
# ./train.sh mbart-en-id-smaller-pre-trained amrbart-new
# ./eval.sh mbart-large-50-finetuned amrbart-new
# ./inference.sh mbart-large-50-finetuned wrete

# ~ Concat
# IS_CONCAT=1 ./train.sh mbart-en-id-smaller-concat-pretrained amrbart-concat
