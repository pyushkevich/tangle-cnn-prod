#!/bin/bash
set -x -e

project_id=${1?}
id=${2?}
svs=${3?}
stain=${4?}
model=${5?}

# Activate service account
gcloud auth activate-service-account --key-file /var/secrets/google/key.json
gcloud config set project $project_id

# Download the SVS
mkdir -p ./data
if ! gsutil cp "gs://mtl_histology/$id/histo_raw/${svs}.*" ./data/; then
  echo "Download of SVS file failed for $id $svs"
  exit -1
fi

# Find the file
svslocal=$(ls ./data/${svs}.*)

# Form the model URL
MODEL_URL="gs://svsbucket/cnn_models/$stain/$model"
if ! gsutil cp "${MODEL_URL}/*" ./model/; then
  echo "Download failed for $MODEL_URL"
  exit -1
fi

# Run the code
python scan_tangles.py apply \
  --slide $svslocal \
  --output ./data/result.nii.gz \
  --network ./model

# Copy result
TRG_URL="gs://mtl_histology/$id/histo_proc/${svs}/density/${svs}_${stain}_${model}_densitymap.nii.gz"
gsutil cp ./data/result.nii.gz $TRG_URL

