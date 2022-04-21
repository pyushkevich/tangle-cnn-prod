#!/bin/bash
# set -x -e

project_id=${1?}
id=${2?}
block=${3?}
sec0=${4-0}
sec1=${5-1000}
force=${6}

# Activate service account
if [[ -f /var/secrets/google/key.json ]]; then
  gcloud auth activate-service-account --key-file /var/secrets/google/key.json
  gcloud config set project $project_id
fi

# Create temporary directory for work
if [[ $TMPDIR ]]; then
  WDIR=$(mktemp -d -t deepcluster-XXXXXX --tmpdir="$TMPDIR")
else
  WDIR=$(mktemp -d -t deepcluster-XXXXXX)
fi

# Get a list of inputs and outputs found
LISTING=$WDIR/alljpg
FINPUTS=$WDIR/inputs
FOUTPUTS=$WDIR/outputs
FNEEDED=$WDIR/needed

# List all blockface images and filter for the ones pertaining to this job
gsutil ls "gs://mtl_histology/$id/bf_raw/${id}_${block}*.jpg" >> $LISTING
rm -rf $FINPUTS
for f in $(cat $LISTING); do
    base=$(basename $f .jpg)
    sec=$(echo $base | awk -F_ '{print $3}')
    sld=$(echo $base | awk -F_ '{print $4}')
    if [[ $sec -ge $sec0 && $sec -lt $sec1 ]]; then
        if [[ $sld -eq 5 || $sld -eq 10 ]]; then
            echo $f >> $FINPUTS
        fi
    fi
done

# List all the available outputs
rm -rf $FOUTPUTS
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

# Are any files needed?
if [[ $(cat $FNEEDED | wc -l) -eq 0 ]]; then
  echo "All required outputs are already present"
  exit 0
fi

# Download the SVS
mkdir -p $WDIR/data $WDIR/model
if ! cat $FNEEDED | gsutil -m cp -I $WDIR/data/; then
  echo "Download of blockface JPEG file failed for $id $block"
  exit 255
fi

# Form the model URL
MODEL_URL="gs://svsbucket/cnn_models/blockface/deepcluster.alexnet.tar"
if ! gsutil -m cp "${MODEL_URL}" $WDIR/model/; then
  echo "Download failed for $MODEL_URL"
  exit 255
fi

# Run the code
for svslocal in $WDIR/data/*.jpg; do
  svs=$(basename $svslocal .jpg)
  python -u blockface_to_multichannel.py apply \
    --slide $svslocal \
    --output $WDIR/data/${svs}_deepcluster.nii.gz \
    --thumb $WDIR/data/${svs}_deepcluster_rgb.nii.gz \
    --network $WDIR/model/deepcluster.alexnet.tar \
    --patch 64 --downsample 16 --batch-size 256
done

# Copy result up to storage
for nii in $WDIR/data/*_deepcluster.nii.gz; do
  svs=$(basename $nii _deepcluster.nii.gz)
  thumb=${nii/.nii.gz/_rgb.nii.gz}
  TRG_URL="gs://mtl_histology/$id/bf_proc/${svs}/preproc/"
  gsutil -m cp $nii $thumb "$TRG_URL"
done

