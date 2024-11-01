#!/bin/bash
set -x -e

project_id=${1?}
id=${2?}
svs=${3?}

# Activate service account
if [[ -f /var/secrets/google/key.json ]]; then
  gcloud auth activate-service-account --key-file /var/secrets/google/key.json
  gcloud config set project $project_id
fi

# Download the SVS
mkdir -p ./data ./model
if ! gsutil -m cp "gs://mtl_histology/$id/bf_raw/${svs}.*" ./data/; then
  echo "Download of blockface JPEG file failed for $id $svs"
  exit 255
fi

# Find the file
svslocal=$(ls ./data/${svs}.*)

# Form the model URL
MODEL_URL="gs://svsbucket/cnn_models/blockface/deepcluster.alexnet.tar"
if ! gsutil -m cp "${MODEL_URL}" ./model/; then
  echo "Download failed for $MODEL_URL"
  exit 255
fi

# Run the code
python -u blockface_to_multichannel.py apply \
  --slide $svslocal \
  --output ./data/result.nii.gz \
  --network ./model/deepcluster.alexnet.tar \
  --patch 64 --downsample 16 --batch-size 256

# Copy result up to storage
TRG_URL="gs://mtl_histology/$id/bf_proc/${svs}/preproc/${svs}_deepcluster.nii.gz"
gsutil cp ./data/result.nii.gz "$TRG_URL"

