#!/bin/python
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.cuda as cutorch
import timeit
import scipy.special
import sys
import openslide
import argparse
import json
import SimpleITK as sitk
import threading
import queue

# Import wildcat models
sys.path.append("wildcat.pytorch")
import wildcat.models

# Import wildcat mods
from unet_wildcat import *

# Set up the device (CPU/GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set up a queue for image chunks
chunk_q = queue.Queue(4)

# Define the worker function to load chunks from the openslide image
def osl_worker(osl, u_range, v_range, chunk_width, overhang):
    for u in range(u_range[0], u_range[1], chunk_width):
        for v in range(v_range[0], v_range[1], chunk_width):

            # Get the width of the region currently covered
            win_dim = np.array((min(u_range[1] - u, chunk_width), min(v_range[1] - v, chunk_width)))

            # Add the overhang - this is what will be read from the image
            win_dim_ohang = win_dim + overhang

            # Read the chunk from the image
            chunk_img=osl.read_region((u,v), 0, win_dim_ohang).convert("RGB")

            # Place into queue
            chunk_q.put(((u,v), win_dim_ohang, chunk_img))

    # Put a sentinel value
    chunk_q.put(None)


def do_info(args):
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    print("CUDA status: ", torch.cuda.is_available())
    print("CUDA memory max alloc: %8.f MB" % (torch.cuda.max_memory_allocated() / 2.0**20))

def read_openslide_chunk(osl, pos, level, size):
    chunk_img=osl.read_region(pos, level, size).convert("RGB")


# Function to apply training to a slide
def do_apply(args):

    # Read the .json config file
    with open(os.path.join(args.network, 'config.json')) as json_file:
        config=json.load(json_file)

    # Read the resnet portion of the config
    cf_resnet = config['resnet']

    # Create the resnet model
    if cf_resnet['size'] == 50:
        model_resnet = models.resnet50(pretrained=False)
    elif cf_resnet['size'] == 18:
        model_resnet = models.resnet18(pretrained=False)
    else:
        raise 'Incompatible resnet model size'

    # Load the resnet model
    model_resnet.fc = nn.Linear(model_resnet.fc.in_features, cf_resnet['num_classes'])
    model_resnet.load_state_dict(torch.load(
        os.path.join(args.network, 'resnet.dat')))
    model_resnet.eval()
    model_resnet = model_resnet.to(device)

    # Read the wildcat portion of the config
    cf_wildcat = config['wildcat_upsample']

    # Read the wildcat model
    model_wildcat = resnet50_wildcat_upsample(
            2, pretrained=False, 
            kmax=cf_wildcat['kmax'],
            alpha=cf_wildcat['alpha'],
            num_maps=cf_wildcat['num_maps'])

    model_wildcat.load_state_dict(
            torch.load(os.path.join(args.network, 'wildcat_upsample.dat')))

    # Set evaluation mode
    model_wildcat.eval()

    # Send model to GPU
    model_wildcat = model_wildcat.to(device)

    # Read the input using OpenSlide
    osl=openslide.OpenSlide(args.slide)

    # The step by which the sampling window is slid (in raw pixels)
    stride = 100

    # Window size in strides (e.g., 400 pixels)
    window_size=4

    # Amount of overhang
    overhang=(window_size-1)*stride

    # Desired number of windows per chunk of SVS loaded into memory at once
    chunk_size=40

    # Actual chunk size (without overhang)
    chunk_width=stride * chunk_size

    # Dimensions of the input image
    slide_dim = np.array(osl.dimensions)

    # Factor by which wildcat shrinks input images when mapping to segmentations
    wildcat_shrinkage=2

    # Scaling factor for output pixels, how many output pixels for every wildcat
    # output pixel. This is needed because for stride of 100, there would be a 
    # non-integer number of output pixels (14/4). 
    out_scale = 1.0 / 8

    # Size of output pixel (in input pixels)
    out_pix_size = wildcat_shrinkage * window_size * stride / (cf_wildcat['input_size'] * out_scale)

    # Output image size 
    out_dim=(slide_dim/out_pix_size).astype(int)

    # Output array (last dimension is per-class probabilities)
    density=np.zeros((2, out_dim[0], out_dim[1]))

    # Range of pixels to scan
    u_range,v_range = (0,slide_dim[0]),(0,slide_dim[1])
    if args.region is not None and len(args.region) == 4:
        R=list(float(val) for val in args.region)
        if all(val < 1.0 for val in R):
            u_range=(int(R[0]*slide_dim[0]),int((R[0]+R[2])*slide_dim[0]))
            v_range=(int(R[1]*slide_dim[1]),int((R[1]+R[3])*slide_dim[1]))
        else:
            u_range=(int(R[0]), int(R[0]+R[2]))
            v_range=(int(R[1]), int(R[1]+R[3]))

    print('Procesing region [%d %d] to [%d %d]' % (u_range[0], v_range[0], u_range[1], v_range[1]))

    # Set up a threaded worker to read openslide pieces
    worker = threading.Thread(target=osl_worker, args=(osl, u_range, v_range, chunk_width, overhang))
    worker.start()

    # Read stuff from queue
    while True:

        # Read a tuple from the queue
        t0 = timeit.default_timer()
        q_data = chunk_q.get()
        t1 = timeit.default_timer()

        # Check for sentinel value
        if q_data is None:
            break

        # Get the offset of the read image, etc
        (u, v) = q_data[0]
        win_dim_ohang = q_data[1]

        # Get the read image
        chunk_img = q_data[2]

        # Resample the chunk for the two networks
        win_dim_ohang_resnet = win_dim_ohang * cf_resnet['input_size'] / (stride * window_size)        
        tran_resnet = transforms.Compose([
            transforms.Resize(int(min(win_dim_ohang_resnet))),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Stride width for resnet (1/4 of 224)
        stride_t = int(cf_resnet['input_size'] / window_size)

        # Convert the read chunk to tensor format
        chunk_tensor=tran_resnet(chunk_img).to(device)
        chunk_ufold=chunk_tensor.unfold(1,cf_resnet['input_size'],stride_t).unfold(2,cf_resnet['input_size'],stride_t).permute((1,2,0,3,4))

        # Create array of windows that are hits
        hits=[]

        # Iterate over windows
        for i in range(0, chunk_ufold.shape[1]):
            for k in range(0, chunk_ufold.shape[0], args.bsr):

                # Process the batch
                k_range = range(k,min(k+args.bsr, chunk_ufold.shape[0]))
                y_ik = model_resnet(chunk_ufold[k_range,i,:,:,:]).cpu().detach().numpy()

                # Record all the hits in this batch
                for j in k_range:
                    if np.argmax(y_ik[j-k,:]) > 0:
                        hits.append([i, j])

        # Finished first pass through the chunk
        t2 = timeit.default_timer()

        # If there are hits, perform wildcat refinement
        if len(hits) > 0:

            # Resample the chunk for the wildcat network
            win_dim_ohang_wildcat = win_dim_ohang * cf_wildcat['input_size'] / (stride * window_size)        
            tran_wildcat = transforms.Compose([
                transforms.Resize(int(min(win_dim_ohang_wildcat))),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

            # Stride width for resnet (1/4 of 448)
            stride_t_wildcat = int(cf_wildcat['input_size'] / window_size)

            # Convert the read chunk to tensor format and unfold
            chunk_tensor_wildcat=tran_wildcat(chunk_img).to(device)
            chunk_ufold_wildcat=chunk_tensor_wildcat.unfold(
                    1,cf_wildcat['input_size'],stride_t_wildcat).unfold(
                            2,cf_wildcat['input_size'],stride_t_wildcat).permute((1,2,0,3,4))

            # Stack the tensors
            hit_folds=[]
            for h in hits:
                hit_folds.append(chunk_ufold_wildcat[h[1],h[0],:,:,:])

            hit_sausage=torch.stack(hit_folds)

            # Process by batches
            for j in range(0, len(hits), args.bsw):
                j_end = min(j+args.bsw, len(hits))

                # Forward pass through the wildcat model
                # x_feat = model_wildcat.features(hit_sausage[j:j_end,:,:,:])
                x_clas = model_wildcat.forward_to_classifier(hit_sausage[j:j_end,:,:,:])
                x_cpool = model_wildcat.spatial_pooling.class_wise(x_clas)

                # Scale the cpool image
                x_cpool_up = torch.nn.functional.interpolate(x_cpool, scale_factor=out_scale)
                w = x_cpool_up.shape[3]

                # Output index for the current hit
                for m in range (j, j_end):
                    u_out = round((u + stride * hits[m][0]) / out_pix_size)
                    v_out = round((v + stride * hits[m][1]) / out_pix_size)
                    density[0, u_out:u_out+w,v_out:v_out+w] += x_cpool_up[m-j,0,:,:].detach().cpu().numpy().transpose()
                    density[1, u_out:u_out+w,v_out:v_out+w] += x_cpool_up[m-j,1,:,:].detach().cpu().numpy().transpose()

        # Finished first pass through the chunk
        t3 = timeit.default_timer()

        # At this point we have a list of hits for this chunk
        print("Chunk: (%6d,%6d) Hits: %4d Times: IO=%6.4f ResN=%6.4f WldC=%6.4f Totl=%8.4f" %
                (u,v,len(hits),t1-t0,t2-t1,t3-t2,t3-t0))

    # Done with thread
    worker.join()

    # Set the spacing based on openslide
    # Get the image spacing from the header, in mm units
    (sx, sy) = (0.0, 0.0)
    if 'openslide.mpp-x' in osl.properties:
        sx = float(osl.properties['openslide.mpp-x']) * out_pix_size / 1000.0
        sy = float(osl.properties['openslide.mpp-y']) * out_pix_size / 1000.0
    elif 'openslide.comment' in osl.properties:
        for z in osl.properties['openslide.comment'].split('\n'):
            r = parse.parse('Resolution = {} um', z)
            if r is not None:
                sx = float(r[0]) * out_pix_size / 1000.0
                sy = float(r[0]) * out_pix_size / 1000.0

    # If there is no spacing, throw exception
    if sx == 0.0 or sy == 0.0:
      raise Exception('No spacing information in image')

    # Report spacing information
    print("Spacing of the mri-like image: %gx%gmm\n" % (sx, sy))

    # Write the result as a NIFTI file
    nii = sitk.GetImageFromArray(np.transpose(density, (2,1,0)), True)
    nii.SetSpacing((sx, sy))
    sitk.WriteImage(nii, args.output)



# Set up an argument parser
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

train_parser = subparsers.add_parser('train')

apply_parser = subparsers.add_parser('apply')
apply_parser.add_argument('--slide', help='Input histology slide to process')
apply_parser.add_argument('--output', help='Where to store the output density map')
apply_parser.add_argument('--network', help='Network saved during training')
apply_parser.add_argument('--bsr', help='Batch size for ResNet', default=8)
apply_parser.add_argument('--bsw', help='Batch size for WildCat', default=8)
apply_parser.add_argument('--region', help='Region of image to process (x,y,w,h)', nargs=4)
apply_parser.set_defaults(func=do_apply)

info_parser = subparsers.add_parser('info')
info_parser.set_defaults(func=do_info)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)
