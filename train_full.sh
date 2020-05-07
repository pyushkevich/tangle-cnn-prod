#!/bin/bash

set -x -e

project_id=${1?}
trainset=${2?}
outmodel=${3?}
epochs=${4?}

# Activate service account
gcloud auth activate-service-account --key-file /var/secrets/google/key.json
gcloud config set project $project_id

# Get training data
mkdir ./data
TRAIN_URL="gs://svsbucket/cnn_train/data/${trainset}.tar"
if ! gsutil cp "${TRAIN_URL}" ./data/; then
  echo "Download failed for $MODEL_URL"
  exit -1
fi

# Unpack the data
pushd data
tar -xvf ./data/${trainset}.tar
popd

# Perform the training
python scan_tangles.py train \
  --datadir ./data/${trainset} --output ./data/model.dat \
  --epochs $epochs --batch 24 

# Upload the model
gsutil cp ./data/model.dat "gs://svsbucket/cnn_train/models/${outmodel}.dat"
