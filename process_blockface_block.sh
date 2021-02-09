#!/bin/bash
#set -x -e

project_id=${1?}
id=${2?}
block=${3?}
force=${4}

# Activate service account
if [[ -f /var/secrets/google/key.json ]]; then
  gcloud auth activate-service-account --key-file /var/secrets/google/key.json
  gcloud config set project $project_id
fi

# Get a list of inputs and outputs found
FINPUTS=$(mktemp /tmp/inputs.XXXXXX)
FOUTPUTS=$(mktemp /tmp/outputs.XXXXXX)
FNEEDED=$(mktemp /tmp/needed.XXXXXX)
for slice in 05 10; do
  gsutil ls "gs://mtl_histology/$id/bf_raw/${id}_${block}_*_${slice}.jpg" >> $FINPUTS
done

if ! gsutil ls "gs://mtl_histology/$id/bf_proc/${id}_${block}_**/*.nii.gz" > $FOUTPUTS; then
  touch $FOUTPUTS
fi

# Filter the inputs
for fn in $(cat $FINPUTS); do
  base=$(basename $fn .jpg)
  if ! grep "${base}_deepcluster.nii.gz" $FOUTPUTS > /dev/null; then 
    echo $fn >> $FNEEDED
  elif ! grep "${base}_deepcluster_rgb.nii.gz" $FOUTPUTS > /dev/null; then 
    echo $fn >> $FNEEDED
  fi
done

# Which file to use
if [[ $force -gt 0 ]]; then
  FNEEDED=$FINPUTS
fi

# Download the SVS
mkdir -p ./data ./model
if ! gsutil -m cp $(cat $FNEEDED) ./data/; then
  echo "Download of blockface JPEG file failed for $id $block"
  exit 255
fi

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

