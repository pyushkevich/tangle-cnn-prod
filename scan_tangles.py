#!/bin/python
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
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
import parse
import traceback

from osl_worker import osl_worker, osl_read_chunk_from_queue

# Import wildcat models
sys.path.append("wildcat.pytorch")
import wildcat.models

# Import wildcat mods
from unet_wildcat import *

# Set up the device (CPU/GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        raise Exception('Incompatible resnet model size')

    # Load the resnet model
    model_resnet.fc = nn.Linear(model_resnet.fc.in_features, cf_resnet['num_classes'])
    model_resnet.load_state_dict(torch.load(
        os.path.join(args.network, 'resnet.dat'),
        map_location=device))
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
            torch.load(
                os.path.join(args.network, 'wildcat_upsample.dat'),
                map_location=device))

    # Read the parameters for scanning
    cf_scan = config['scan']

    # Set evaluation mode
    model_wildcat.eval()

    # Send model to GPU
    model_wildcat = model_wildcat.to(device)

    # Read the input using OpenSlide
    osl=openslide.OpenSlide(args.slide)
    slide_dim = np.array(osl.dimensions)

    # Size of the training patch used to train wildcat, in raw pixels
    patch_size_raw = cf_scan.get('patch_size_raw', 512)

    # Size to which these patches are downsampled
    input_size_wildcat = cf_wildcat['input_size']

    # Size of the window used to apply WildCat. Should be larger than the patch size
    # This does not include the padding
    window_size_raw = cf_scan.get('window_size_raw', 4096)

    # The amount of padding, relative to patch size to add to the window. This padding
    # is to provide context at the edges of the window
    padding_size_rel = cf_scan.get('padding_size_rel', 1.0)
    padding_size_raw = int(padding_size_rel * patch_size_raw)

    # Factor by which wildcat shrinks input images when mapping to segmentations
    wildcat_shrinkage=cf_wildcat.get('shrinkage', 2)

    # Additional shrinkage to apply to output (because we don't want to store very large)
    # output images
    extra_shrinkage=cf_scan.get('extra_shrinkage', 4)

    # Size of output pixel (in input pixels)
    out_pix_size = wildcat_shrinkage * extra_shrinkage * patch_size_raw * 1.0 / input_size_wildcat

    # The output size for each window
    window_size_out = int(window_size_raw / out_pix_size)

    # The padding size for the output
    padding_size_out = int(padding_size_rel * patch_size_raw / out_pix_size)

    # Total number of non-overlapping windows to process
    n_win = np.ceil(slide_dim / window_size_raw).astype(int)

    # Output image size 
    out_dim=(n_win * window_size_out).astype(int)

    # Output array (last dimension is per-class probabilities)
    density=np.zeros((2, out_dim[0], out_dim[1]))

    # Range of pixels to scan
    u_range,v_range = (0,n_win[0]),(0,n_win[1])

    # Allow a custom region to be specified
    if args.region is not None and len(args.region) == 4:
        R=list(float(val) for val in args.region)
        if all(val < 1.0 for val in R):
            u_range=(int(R[0]*n_win[0]),int((R[0]+R[2])*n_win[0]))
            v_range=(int(R[1]*n_win[1]),int((R[1]+R[3])*n_win[1]))
        else:
            u_range=(int(R[0]), int(R[0]+R[2]))
            v_range=(int(R[1]), int(R[1]+R[3]))

    print('Procesing region [%d %d] to [%d %d]' % (u_range[0], v_range[0], u_range[1], v_range[1]))

    # Set up a threaded worker to read openslide patches
    worker = threading.Thread(target=osl_worker, args=(osl, u_range, v_range, window_size_raw, padding_size_raw))
    worker.start()

    # Try/catch block to kill worker when done
    t_00 = timeit.default_timer()
    try:
        
        # Range non-overlapping windows
        while True:

            # Read the chunk from the image
            t0 = timeit.default_timer()
            q_data = osl_read_chunk_from_queue()
            t1 = timeit.default_timer()

            # Check for sentinel value
            if q_data is None:
                break

            # Get the values
            ((u,v), (x,y,w), (xp,yp,wp), chunk_img) = q_data
                    
            # Compute the desired size of input to wildcat
            wwc = int(wp * input_size_wildcat / patch_size_raw)

            # Resample the chunk for the two networks
            tran = transforms.Compose([
                transforms.Resize((wwc,wwc)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Convert the read chunk to tensor format
            with torch.no_grad():
                
                # Apply transforms and turn into correct-size torch tensor
                chunk_tensor=torch.unsqueeze(tran(chunk_img),dim=0).to(device)
                
                # Forward pass through the wildcat model
                x_clas = model_wildcat.forward_to_classifier(chunk_tensor)
                x_cpool = model_wildcat.spatial_pooling.class_wise(x_clas)

                # Scale the cpool image to desired size
                x_cpool_up = torch.nn.functional.interpolate(x_cpool, scale_factor=1.0/extra_shrinkage).detach().cpu().numpy()

                # Extract the central portion of the output
                p0,p1 = padding_size_out,(padding_size_out+window_size_out)
                x_cpool_ctr = x_cpool_up[:,:,p0:p1,p0:p1]
                
                # Stick it into the output array
                xout0,xout1 = u * window_size_out, ((u+1) * window_size_out)
                yout0,yout1 = v * window_size_out, ((v+1) * window_size_out)
                density[0,xout0:xout1,yout0:yout1] = x_cpool_ctr[0,0,:,:].transpose()
                density[1,xout0:xout1,yout0:yout1] = x_cpool_ctr[0,1,:,:].transpose()
                    
            # Finished first pass through the chunk
            t2 = timeit.default_timer()
            
            # At this point we have a list of hits for this chunk
            print("Chunk: (%6d,%6d) Times: IO=%6.4f WldC=%6.4f Totl=%8.4f" %
                  (u,v,t1-t0,t2-t1,t2-t0))
            
        # Trim the density array to match size of input
        out_dim_trim=np.round((slide_dim/out_pix_size)).astype(int)
        density=density[:,0:out_dim_trim[0],0:out_dim_trim[1]]

        # Report total time
        t_11 = timeit.default_timer()
        print("Total time elapsed: %8.4f" % (t_11-t_00,))

    except:
        traceback.print_exc()
        sys.exit(-1)

    finally:
        worker.join(60)
        if worker.isAlive():
            print('Thread worker failed to terminate after 60 seconds')

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



# Standard training code
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        estart = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels_one_hot = torch.zeros([labels.shape[0], 2])
                labels_one_hot[:,0] = (labels==0)
                labels_one_hot[:,1] = (labels==1)
                labels_one_hot = labels_one_hot.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_one_hot)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f} Time: {:8.4f}'.format(phase, epoch_loss, epoch_acc, (time.time()-estart)))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def do_train(args):

    # Number of classes in the dataset
    num_classes = 2

    # Batch size for training (change depending on how much memory you have)
    batch_size = int(args.batch)
    num_epochs = int(args.epochs)

    # Input image size
    input_size = 224

    # Initialize the wildcat model
    model=resnet50_wildcat_upsample(2, pretrained=True, kmax=0.02, kmin=0.0, alpha=0.7, num_maps=4)

    # Loss and optimizer
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.SGD(model.get_config_optim(0.01, 0.1), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # Transforms for training and validation
    data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomRotation(45),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }

    # Training and validation dataloaders
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.datadir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Map to CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    model = model.to(device)
    criterion = criterion.to(device)

    # Run the training
    model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

    # After the training, save the model
    torch.save(model_ft.state_dict(), args.output)



# Set up an argument parser
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

train_parser = subparsers.add_parser('train')
train_parser.add_argument('--datadir', help='source directory')
train_parser.add_argument('--output', help='output classifier')
train_parser.add_argument('--epochs', help='number of epochs')
train_parser.add_argument('--batch', help='batch size')
train_parser.set_defaults(func=do_train)


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
