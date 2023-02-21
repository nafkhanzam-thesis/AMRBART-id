#! /bin/bash

git clone https://github.com/nafkhanzam-thesis/AMRBART-v3
cd AMRBART-v3

apt install git-lfs
git lfs install

mkdir models
pushd models
git clone https://huggingface.co/sshleifer/tiny-mbart
popd

mkdir outputs

pip install -r requirements.txt
