# Set up
import os
import sys
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import openslide
import math
import argparse
import parse
import SimpleITK as sitk
import threading
from osl_worker import osl_worker, osl_read_chunk_from_queue

# Deepcluster
sys.path.append("deepcluster")
from deepcluster.util import load_model


def run_model_on_window(model, W):
    w_patch = W.unfold(1, 256, 256).unfold(2, 256, 256).permute(1, 2, 0, 3, 4).reshape(-1, 3, 256, 256)
    k = int(math.sqrt(w_patch.shape[0]))
    w_patch_crop = w_patch[:, :, 16:240, 16:240]
    with torch.no_grad():
        w_result = model(w_patch_crop)
    return w_result.reshape(k, k, w_result.shape[1])


# Process a slide
def apply_to_slide(args):
    # Read the trained model
    model_file = os.path.abspath('./exp01/checkpoint.200th.tar')
    model = load_model(args.network)
    model.cuda()
    model.eval()

    # Read the slide
    osl = openslide.OpenSlide(args.slide)

    # Transform
    tran_norm = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Break the image into patches, for each patch, run the model on it
    (w, h) = osl.level_dimensions[0]

    # Size of the input window and number of windows
    window_size = int(args.window)
    nw_x, nw_y = math.ceil(w / window_size), math.ceil(h / window_size)

    # Output window size and output dimensions
    output_window_size = int(window_size / 256)
    (ow, oh) = nw_x * output_window_size, nw_y * output_window_size
    output = np.zeros((ow, oh, 20))

    # Set up a threaded worker to read openslide patches
    worker = threading.Thread(target=osl_worker, args=(osl, (0, nw_x), (0, nw_y), window_size, 0))
    worker.start()

    while True:

        # Read a chunk of data
        q_data = osl_read_chunk_from_queue()

        # Check for sentinel value
        if q_data is None:
            break

        # Get the values
        ((i_x, i_y), (c_x, c_y, wd), _, window) = q_data

        # The corner of the region
        W = tran_norm(window).cuda()
        R = run_model_on_window(model, W).cpu().numpy().transpose((1, 0, 2))

        co_x, co_y = output_window_size * i_x, output_window_size * i_y
        output[co_x:co_x + output_window_size, co_y:co_y + output_window_size, :] = R
        print('Finished (%d,%d) of (%d,%d)' % (i_x, i_y, nw_x, nw_y))

    # Clip the output
    output = output[0:math.ceil(w / 256), 0:math.ceil(h / 256), :].transpose(1, 0, 2)

    # Set the spacing based on openslide
    # Get the image spacing from the header, in mm units
    (sx, sy) = (0.0, 0.0)
    if 'openslide.mpp-x' in osl.properties:
        sx = float(osl.properties['openslide.mpp-x']) * 256 / 1000.0
        sy = float(osl.properties['openslide.mpp-y']) * 256 / 1000.0
    elif 'openslide.comment' in osl.properties:
        for z in osl.properties['openslide.comment'].split('\n'):
            r = parse.parse('Resolution = {} um', z)
            if r is not None:
                sx = float(r[0]) * 256 / 1000.0
                sy = float(r[0]) * 256 / 1000.0

    # If there is no spacing, throw exception
    if sx == 0.0 or sy == 0.0:
        raise Exception('No spacing information in image')

    # Report spacing information
    print("Spacing of the mri-like image: %gx%gmm\n" % (sx, sy))

    # Write the result as a NIFTI file
    nii = sitk.GetImageFromArray(np.transpose(output, (0, 1, 2)), True)
    nii.SetSpacing((sx, sy))
    sitk.WriteImage(nii, args.output)


# Set up an argument parser
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

apply_parser = subparsers.add_parser('apply')
apply_parser.add_argument('--slide', help='Input histology slide to process')
apply_parser.add_argument('--output', help='Where to store the output density map')
apply_parser.add_argument('--network', help='Network saved during training')
apply_parser.add_argument('--window', help='Window size', default=4096)
apply_parser.add_argument('--region', help='Region of image to process (x,y,w,h)', nargs=4)
apply_parser.set_defaults(func=apply_to_slide)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)
