#!/bin/bash
set -x -e

project_id=${1?}
id=${2?}
svs=${3?}

# Activate service account
gcloud auth activate-service-account --key-file /var/secrets/google/key.json
gcloud config set project $project_id

# Download the SVS
mkdir -p ./data ./model
if ! gsutil cp "gs://mtl_histology/$id/histo_raw/${svs}.*" ./data/; then
  echo "Download of SVS file failed for $id $svs"
  exit 255
fi

# Find the file
svslocal=$(ls ./data/${svs}.*)

# Make sure the file is multi-level
LEVELS=$(python levels.py $svslocal | awk '{print $4}')
if [[ $LEVELS -le 1 ]]; then
  
  mkdir -p ./data/fixflat
  mv $svslocal ./data/fixflat
  svsflat=$(ls ./data/fixflat/${svs}.*)
  vips tiffsave $svsflat $svslocal \
    --vips-progress --compression=jpeg --Q=80 \
    --tile --tile-width=256 --tile-height=256 \
    --pyramid --bigtiff

fi

# Form the model URL
MODEL_URL="gs://svsbucket/cnn_models/nissl/deepcluster.alexnet.tar"
if ! gsutil cp "${MODEL_URL}/*" ./model/; then
  echo "Download failed for $MODEL_URL"
  exit 255
fi

# Run the code
python -u nissl_to_multichannel.py apply \
  --slide $svslocal \
  --output ./data/result.nii.gz \
  --network ./model/deepcluster.alexnet.tar

# Copy result up to storage
for ext in 'nii.gz' 'tiff'; do
  TRG_URL="gs://mtl_histology/$id/histo_proc/${svs}/preproc/${svs}_deepcluster.nii.gz"
  gsutil cp ./data/result.nii.gz ${TRG_URL}
done

