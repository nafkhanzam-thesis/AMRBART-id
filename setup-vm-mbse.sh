#! /bin/bash

#~ Variables

export model_name=amr3joint_ontowiki2_g2g-structured-bart-large

#~ Install MBSE Parser

git clone https://github.com/nafkhanzam-thesis/transition-amr-parser.git
cd transition-amr-parser

pip install .
python setup.py install

#~ Download MBSE Model

pip install awscli
sudo apt install unzip

# Assign AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY Environment Variables

aws s3 cp --endpoint-url https://s3.us-east.cloud-object-storage.appdomain.cloud s3://mnlp-models-amr/$model_name.zip .
unzip $model_name.zip

#~ Download Dataset

wget https://storage.nafkhanzam.com/thesis/backups/amrbart-tnp.tar.gz
tar -xvzf amrbart-tnp.tar.gz

#~ Parse

amr-parse --tokenize -c $model_name/checkpoint_wiki.smatch_top5-avg.pt -i amrbart-tnp/LDC2017-test.en-trans -o LDC2017-test.mbse.amr
