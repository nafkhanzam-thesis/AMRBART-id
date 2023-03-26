#! /bin/bash

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
# export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
# export CARGO_HTTP_MULTIPLEXING=false

git clone https://github.com/nafkhanzam-thesis/AMRBART-v3
cd AMRBART-v3/wikification

pip install -r requirements.txt

git clone https://github.com/facebookresearch/BLINK.git
pushd BLINK
  # tail -n +2 requirements.txt > requirements.txt.output
  # mv requirements.txt.output requirements.txt
  # pip install -r requirements.txt
  chmod +x download_blink_models.sh
  ./download_blink_models.sh
popd

mkdir data
