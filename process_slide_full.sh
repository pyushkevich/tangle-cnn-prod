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
mkdir -p ./data ./model
if ! gsutil cp "gs://mtl_histology/$id/histo_raw/${svs}.*" ./data/; then
  echo "Download of SVS file failed for $id $svs"
  exit -1
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
MODEL_URL="gs://svsbucket/cnn_models/$stain/$model"
if ! gsutil cp "${MODEL_URL}/*" ./model/; then
  echo "Download failed for $MODEL_URL"
  exit -1
fi

# Run the code
python -u scan_tangles.py apply \
  --slide $svslocal \
  --output ./data/result.nii.gz \
  --network ./model

# Convert the result to pyramidal tiff (for visualization)
c3d -mcs ./data/result.nii.gz \
  -scale -1 -add -scale -1 -info -stretch -127.5 127.5 0 255 -clip 0 255 \
  -type uchar -o ./data/result_flat.tiff

vips tiffsave ./data/result_flat.tiff ./data/result.tiff \
  --vips-progress --compression=deflate \
  --tile --tile-width=256 --tile-height=256 \
  --pyramid --bigtiff

# Copy result
for ext in 'nii.gz' 'tiff'; do
  TRG_URL="gs://mtl_histology/$id/histo_proc/${svs}/density/${svs}_${stain}_${model}_densitymap"
  gsutil cp ./data/result.${ext} ${TRG_URL}.${ext}
done

