#!/bin/bash
set -x -e

project_id=${1?}
id=${2?}
block=${3?}

# Activate service account
if [[ -f /var/secrets/google/key.json ]]; then
  gcloud auth activate-service-account --key-file /var/secrets/google/key.json
  gcloud config set project $project_id
fi

# Download the SVS
mkdir -p ./data ./model
for slice in 05 10; do
  if ! gsutil -m cp "gs://mtl_histology/$id/bf_raw/${id}_${block}_*_${slice}.jpg" ./data/; then
    echo "Download of blockface JPEG file failed for $id $block"
    exit 255
  fi
done

# Form the model URL
MODEL_URL="gs://svsbucket/cnn_models/blockface/deepcluster.alexnet.tar"
if ! gsutil -m cp "${MODEL_URL}" ./model/; then
  echo "Download failed for $MODEL_URL"
  exit 255
fi

# Run the code
for svslocal in ./data/*.jpg; do
  svs=$(basename $svslocal .jpg)
  python -u blockface_to_multichannel.py apply \
    --slide $svslocal \
    --output ./data/${svs}_deepcluster.nii.gz \
    --thumb ./data/${svs}_deepcluster_rgb.nii.gz \
    --network ./model/deepcluster.alexnet.tar \
    --patch 64 --downsample 16 --batch-size 256

  break
done

# Copy result up to storage
for nii in ./data/*_deepcluster.nii.gz; do
  svs=$(basename $nii _deepcluster.nii.gz)
  thumb=${nii/.nii.gz/_rgb.nii.gz}
  TRG_URL="gs://mtl_histology/$id/bf_proc/${svs}/preproc/"
  gsutil -m cp $nii $thumb "$TRG_URL"
done
