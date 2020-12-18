#!/bin/bash
set -x -e

function usage()
{
cat << 'USAGE'
wildcat_analyze_wsi.sh: Script to apply trained WildCat model to whole slide
usage:
  wildcat_analyze_wsi.sh [options]
required options:
  -s <file|url>              Path or gs:// URL to the input whole-slide image
  -n <file|url>              Path or gs:// URL to the folder containing trained network
  -o <file|url>              Path or gs:// URL to the output density .nii.gz file
additional options:
  -p <project_name>          Specify GCP project id (for GCP-stored data)
  -k <key_json>              Specify the secret key for desired GCP service account
  -t <pattern>               Also generate per-label heat maps in pyramid tiff format. The pattern
                             should include a %s string that will be replaced with the label name,
                             e.g., gs://my_bucket/spec01/slide35/density/nissl_%s_density.tiff
  -D <scalar>                Downsampling factor for the density map. The downsampling is relative to
                             the WildCat output (1/2 of the size of the input image). The default is 4
                             but this might generate huge images, so 8 or 16 may be recommended
  -W <scalar>                Window size for WildCat slide scanning. Default is 2048 which works ok
                             on machines with 16GB GPU memory, but smaller memory footprint may require
                             a smaller window size
USAGE
}

# Some parameters must be supplied
if [ $# -lt 1 ]; then
  usage
  return 255
fi

# Variables used in the script
unset SLIDE_URL NETWORK_URL DENSITY_URL PROJECT_ID PROJECT_KEY TIFF_PATTERN
DOWNSAMPLING=4
WINDOWSIZE=2048

while getopts ":s:n:o:p:k:t:D:W:" opt; do
  case ${opt} in
    s ) SLIDE_URL=$OPTARG;;
    n ) NETWORK_URL=$OPTARG;;
    o ) DENSITY_URL=$OPTARG;;
    p ) PROJECT_ID=$OPTARG;;
    k ) PROJECT_KEY=$OPTARG;;
    t ) TIFF_PATTERN=$OPTARG;;
    D ) DOWNSAMPLING=$OPTARG;;
    W ) WINDOWSIZE=$OPTARG;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done
shift $((OPTIND -1))

# Activate GCP if provided
if [[ $PROJECT_KEY ]]; then
  gcloud auth activate-service-account --key-file "$PROJECT_KEY"
fi
if [[ $PROJECT_ID ]]; then
  gcloud config set project "$PROJECT_ID"
fi

# Create temporary directory for work
if [[ $TMPDIR ]]; then
  WDIR=$(mktemp -d -t wildcat-XXXXXX --tmpdir="$TMPDIR")
else
  WDIR=$(mktemp -d -t wildcat-XXXXXX)
fi

# Get the components of the slide filename
SLIDE_BN=$(basename "$SLIDE_URL")
SLIDE_BN_NOEXT=${SLIDE_BN/\..*$/\.tiff}

# Download the input file if needed
if [[ "${SLIDE_URL?}" =~ gs:// ]]; then
  gsutil -m cp "$SLIDE_URL" "$WDIR/" || (echo "Unable to download $SLIDE_URL" && exit 255)
  SLIDE_URL=$WDIR/$SLIDE_BN
fi

# Download the model if needed
if [[ "${NETWORK_URL?}" =~ gs:// ]]; then
  mkdir -p "$WDIR/model"
  gsutil cp "$NETWORK_URL/*" "$WDIR/model/" || echo "Unable to download $NETWORK_URL" && exit 255
  NETWORK_URL=$WDIR/model
fi

# Set up the output
if [[ "${DENSITY_URL?}" =~ gs:// ]]; then
  DENSITY_NII=$WDIR/$(basename "$DENSITY_URL")
else
  DENSITY_NII=$DENSITY_URL
fi

# Make sure the file is multi-level
LEVELS=$(python levels.py "$SLIDE_URL" | awk '{print $4}')
if [[ $LEVELS -le 1 ]]; then
  mkdir -p "$WDIR/fixflat"
  SLIDE_FIXFLAT="$WDIR/fixflat/$SLIDE_BN_NOEXT.tiff"
  vips tiffsave "$SLIDE_URL" "$SLIDE_FIXFLAT" \
    --vips-progress --compression=jpeg --Q=80 \
    --tile --tile-width=256 --tile-height=256 \
    --pyramid --bigtiff

  SLIDE_URL=$SLIDE_FIXFLAT
fi

# Perform the scanning in Python
python -u wildcat_main.py apply \
  --modeldir "$NETWORK_URL" \
  --slide "$SLIDE_URL" \
  --output "$DENSITY_NII" \
  --shrink "$DOWNSAMPLING" \
  --window "$WINDOWSIZE"

# Send the output to GCP if needed
if [[ "${DENSITY_URL?}" =~ gs:// ]]; then
  gsutil -m cp "$DENSITY_NII" "$DENSITY_URL" || echo "Unable to upload $DENSITY_URL" && exit 255
fi

