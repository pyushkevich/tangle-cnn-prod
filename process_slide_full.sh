#!/bin/bash
PROJECT_ID=${1?}
SRC_URL=${2?}
TRG_URL=${3?}
MODEL_URL=${4?}

# Activate service account
gcloud auth activate-service-account --key-file /var/secrets/google/key.json

# Set default project
gcloud config set project $PROJECT_ID

# Copy the model into desired location
mkdir -p model data
gsutil cp -r $MODEL_URL ./model/

# Copy the data to the data location
gsutil cp $SRC_URL ./data
SLIDE=./data/$(ls -tr ./data | tail -n 1)

# Run the code
python scan_tangles.py apply --slide $SLIDE --output ./data/result.nii.gz \
  --network ./model/model_wildcat_upsample 

# Upload the result
gsutil cp ./data/result.nii.gz $TRG_URL

