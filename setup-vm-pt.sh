#! /bin/bash -i

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda install pytorch=*=*cuda* cudatoolkit -c pytorch -y

git clone https://github.com/nafkhanzam-thesis/AMRBART-v3
cd AMRBART-v3

mkdir models
pushd models
  #~ mBART-large-50
  # sudo apt install git-lfs
  # git lfs install
  # git clone https://huggingface.co/facebook/mbart-large-50

  #~ mBART-en-id
  wget https://storage.nafkhanzam.com/thesis/backups/mbart-en-id-smaller.tar.gz
  tar -xvzf mbart-en-id-smaller.tar.gz
popd

mkdir datasets
pushd datasets
  wget https://storage.nafkhanzam.com/thesis/ds.tar.xz
  tar xvJf ds.tar.xz
  wget https://storage.nafkhanzam.com/thesis/backups/amrbart-datasets.tar.gz
  tar -xvzf amrbart-datasets.tar.gz
popd

pip install -r requirements.txt

# python main.py mbart-large-50 gpu 2>&1 | tee ../outputs/mbart-large-50/run.log
# python main.py mbart-en-id-smaller gpu 2>&1 | tee ../outputs/mbart-en-id-smaller/run.log
