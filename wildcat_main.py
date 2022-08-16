#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import timeit
import scipy.special
import sys
import openslide
import argparse
import json
import SimpleITK as sitk
import threading
import parse
import glob
import traceback
import matplotlib.pyplot as plt
from osl_worker import osl_worker, osl_read_chunk_from_queue
import wildcat_pytorch.wildcat as wildcat
from wildcat_pytorch.picsl_wildcat import models, util
from unet_wildcat_gmm import UNet_WSL_GMM
from PIL import Image
import pandas as pd

# Set up the device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def do_info(dummy_args):
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    print("CUDA status: ", torch.cuda.is_available())
    print("CUDA memory max alloc: %8.f MB" % (torch.cuda.max_memory_allocated() / 2.0 ** 20))


def read_openslide_chunk(osl, pos, level, size):
    chunk_img = osl.read_region(pos, level, size).convert("RGB")


def make_model(config):
    """Instantiate a model, loss, and optimizer based on the config dict"""
    mcfg = config['wildcat_upsample']

    # Instantiate WildCat model, loss and optiizer
    if mcfg['gmm'] > 0:
        model = UNet_WSL_GMM(
            num_classes=config['num_classes'],
            mix_per_class=mcfg['num_maps'],
            kmax=mcfg['kmax'],
            kmin=mcfg['kmin'],
            alpha=mcfg['alpha'],
            gmm_iter=mcfg['gmm']
        )

        # We use BCE loss because network outputs are probabilities, and this loss
        # does log clamping to prevent infinity or NaN in the gradients
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-2)

    else:
        model = models.resnet50_wildcat_upsample(
            config['num_classes'],
            pretrained=True,
            kmax=mcfg['kmax'],
            kmin=mcfg['kmin'],
            alpha=mcfg['alpha'],
            num_maps=mcfg['num_maps'])

        # Loss and optimizer
        criterion = nn.MultiLabelSoftMarginLoss()
        optimizer = torch.optim.SGD(model.get_config_optim(0.01, 0.1), lr=0.01, momentum=0.9, weight_decay=1e-2)

    return model, criterion, optimizer


# Function to apply training to a slide
def do_apply(args):

    # Set the model directory
    model_dir = args.modeldir

    # Read the model's .json config file
    with open(os.path.join(model_dir, 'config.json')) as json_file:
        config = json.load(json_file)

    # Create the model
    model_ft, _, _ = make_model(config)

    # Read model state
    model_ft.load_state_dict(
        torch.load(os.path.join(model_dir, "wildcat_upsample.dat")))

    # Send model to GPU
    model_ft.eval()
    model_ft = model_ft.to(device)

    # Read the input using OpenSlide
    osl = openslide.OpenSlide(args.slide)
    slide_dim = np.array(osl.dimensions)

    # Input size to WildCat (should be 224)
    input_size_wildcat = config['wildcat_upsample']['input_size']

    # Corresponding patch size in raw image (should match WildCat, no downsampling)
    patch_size_raw = config['wildcat_upsample']['input_size']

    # Size of the window used to apply WildCat. Should be larger than the patch size
    # This does not include the padding
    window_size_raw = int(args.window)

    # The amount of padding, relative to patch size to add to the window. This padding
    # is to provide context at the edges of the window
    padding_size_rel = 1.0
    padding_size_raw = int(padding_size_rel * patch_size_raw)

    # Factor by which wildcat shrinks input images when mapping to segmentations
    wildcat_shrinkage = 2

    # Additional shrinkage to apply to output (because we don't want to store very large)
    # output images
    extra_shrinkage = int(args.shrink)

    # Size of output pixel (in input pixels)
    out_pix_size = wildcat_shrinkage * extra_shrinkage * patch_size_raw * 1.0 / input_size_wildcat

    # The output size for each window
    window_size_out = int(window_size_raw / out_pix_size)

    # The padding size for the output
    padding_size_out = int(padding_size_rel * patch_size_raw / out_pix_size)

    # Total number of non-overlapping windows to process
    n_win = np.ceil(slide_dim / window_size_raw).astype(int)

    # Output image size 
    out_dim = (n_win * window_size_out).astype(int)

    # Output array (last dimension is per-class probabilities)
    num_classes = config['num_classes']
    density = np.zeros((num_classes, out_dim[0], out_dim[1]))

    # Range of pixels to scan
    u_range, v_range = (0, n_win[0]), (0, n_win[1])

    # Allow a custom region to be specified
    if args.region is not None and len(args.region) == 4:
        region = list(float(val) for val in args.region)
        if all(val < 1.0 for val in region):
            u_range = (int(region[0] * n_win[0]), int((region[0] + region[2]) * n_win[0]))
            v_range = (int(region[1] * n_win[1]), int((region[1] + region[3]) * n_win[1]))
        else:
            u_range = (int(region[0]), int(region[0] + region[2]))
            v_range = (int(region[1]), int(region[1] + region[3]))

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
            ((u, v), (x, y, w), (xp, yp, wp), chunk_img) = q_data

            # Compute the desired size of input to wildcat
            wwc = int(wp * input_size_wildcat / patch_size_raw)

            # Resample the chunk for the two networks
            tran = transforms.Compose([
                transforms.Resize((wwc, wwc)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # Convert the read chunk to tensor format
            with torch.no_grad():

                # Apply transforms and turn into correct-size torch tensor
                chunk_tensor = torch.unsqueeze(tran(chunk_img), dim=0).to(device)

                # Forward pass through the wildcat model
                x_clas = model_ft.forward_to_classifier(chunk_tensor)
                x_cpool = model_ft.spatial_pooling.class_wise(x_clas)

                # Scale the cpool image to desired size
                x_cpool_up = torch.nn.functional.interpolate(x_cpool,
                                                             scale_factor=1.0 / extra_shrinkage).detach().cpu().numpy()

                # Extract the central portion of the output
                p0, p1 = padding_size_out, (padding_size_out + window_size_out)
                x_cpool_ctr = x_cpool_up[:, :, p0:p1, p0:p1]

                # Stick it into the output array
                xout0, xout1 = u * window_size_out, ((u + 1) * window_size_out)
                yout0, yout1 = v * window_size_out, ((v + 1) * window_size_out)
                for j in range(num_classes):
                    density[j, xout0:xout1, yout0:yout1] = x_cpool_ctr[0, j, :, :].transpose()

            # Finished first pass through the chunk
            t2 = timeit.default_timer()

            # At this point we have a list of hits for this chunk
            print("Chunk: (%6d,%6d) Times: IO=%6.4f WldC=%6.4f Totl=%8.4f" %
                  (u, v, t1 - t0, t2 - t1, t2 - t0))

        # Trim the density array to match size of input
        out_dim_trim = np.round((slide_dim / out_pix_size)).astype(int)
        density = density[:, 0:out_dim_trim[0], 0:out_dim_trim[1]]

        # Report total time
        t_11 = timeit.default_timer()
        print("Total time elapsed: %8.4f" % (t_11 - t_00,))

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
    nii_data = np.transpose(density, (2, 1, 0))
    print('Output data shape: ', nii_data.shape)
    nii = sitk.GetImageFromArray(nii_data, True)
    print('Setting spacing to', (sx, sy))
    nii.SetSpacing((sx, sy))
    print("Density map will be saved to ", args.output)
    sitk.WriteImage(nii, args.output)


# Function to apply training to a collection of extracted patches
def do_patch_apply(args):

    # Set the model directory
    model_dir = args.modeldir

    # Read the model's .json config file
    with open(os.path.join(model_dir, 'config.json')) as json_file:
        config = json.load(json_file)

    # Number of classes in the model
    num_classes = config['num_classes']

    # Dict storing model statistics
    d = { 'patches' : [], 'means': [], 'histograms': [] }

    # Create the model
    model_ft, _, _ = make_model(config)

    # Read model state
    model_ft.load_state_dict(
        torch.load(os.path.join(model_dir, "wildcat_upsample.dat")))

    # Send model to GPU
    model_ft.eval()
    model_ft = model_ft.to(device)
    print("Model loaded to device")

    # Find all the input files
    allstat = {}

    # Read all filenames to process
    fn_list = glob.glob(os.path.join(args.input, "*.png"))
    if len(fn_list) == 0:
        print('Missing input files')
        return 255

    # Split into mini-batch chunks
    fn_list_fold = [fn_list[i:i+args.batch] for i in range(0, len(fn_list), args.batch)]

    # Read the first patch to determine the patch size
    patch = Image.open(fn_list[0]).convert("RGB")

    # Set up the transformation
    pw,ph = int(patch.size[0] * args.scale),int(patch.size[1] * args.scale)
    tran = transforms.Compose([
        transforms.Resize((pw,ph)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Set up the minibatch storage
    chunk_tensor = torch.zeros(args.batch, 3, pw, ph).to(device)

    # Process data in minibatches
    for ib, b in enumerate(fn_list_fold):
        print('Batch {} of {}'.format(ib, len(fn_list_fold)))
        
        # Load patches from images, transform, and stick into tensor
        for ifn, fn in enumerate(b):
            if ib > 0 and ifn > 0:
                patch = Image.open(fn).convert("RGB")
            chunk_tensor[ifn,:] = tran(patch).to(device)

        # Process minibatch through model
        with torch.no_grad():
            x_clas = model_ft.forward_to_classifier(chunk_tensor)
            x_cpool = model_ft.spatial_pooling.class_wise(x_clas)

        # Collect statistics
        for ifn, fn in enumerate(b):
            # Get statistics for each class 
            z = x_cpool.cpu().detach().numpy()
            zstat = []
            for k in range(num_classes):
                zk = z[0,k,:,:].flatten()
                zhist = np.histogram(zk, bins=np.linspace(-6.0, 6.0, 121), density=True)
                zmean = np.mean(zk)
                zstd = np.std(zk)
                zstat.append({'histogram': zhist[0].tolist(), 'mean': zmean.item(), 'std': zstd.item()})
            
            # Append the statistics
            allstat[os.path.basename(fn)] = zstat

            # Write the image back to destination dir if requested
            if args.outdir:
                fn_dest = os.path.join(args.outdir, "wildcat_" + os.path.splitext(os.path.basename(fn))[0] + ".nii.gz")
                dimg = x_cpool[0,:,:,:].cpu().numpy().transpose(1,2,0)
                nii = sitk.GetImageFromArray(dimg, True)
                sitk.WriteImage(nii, fn_dest)

    # Write the output JSON file, if requested
    if args.outstat:
        with open(args.outstat, "w") as f_outstat:
            json.dump(allstat, f_outstat)


# Standard training code
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    train_acc_history = []
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Number of classes
    num_classes = len(dataloaders['train'].dataset.class_to_idx)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            nmb = len(dataloaders[phase])
            for mb, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels_one_hot = torch.zeros([labels.shape[0], num_classes])
                for i in range(num_classes):
                    labels_one_hot[:, i] = (labels == i)
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

                # Print minimatch stats
                print('MB %04d/%04d  loss %f  corr %d' %
                      (mb, nmb, loss.item(), torch.sum(preds == labels.data).item()))

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc.item())
            elif phase == 'train':
                train_acc_history.append(epoch_acc.item())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history


def do_train(arg):
    global device

    # Read the experiment directory
    exp_dir = arg.expdir

    # Save the model using a format that can be read using production-time scripts
    model_dir = os.path.join(exp_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # The location of the config file
    config_file = os.path.join(model_dir, 'config.json')
    model_file = os.path.join(model_dir, 'wildcat_upsample.dat')

    # Are we resuming - then load old parameters
    if bool(arg.resume) is True:
        print('Loading config parameters from', config_file)
        with open(config_file, 'r') as jfile:
            config = json.load(jfile)
            hist_train = config['train_history']['train']
            hist_val = config['train_history']['val']
    else:
        # Set up the configuration
        config = {
            "wildcat_upsample": {
                "kmax": float(arg.kmax),
                "kmin": float(arg.kmin),
                "alpha": float(arg.alpha),
                "num_maps": int(arg.nmaps),
                "input_size": 224,
                "num_epochs": int(arg.epochs),
                "batch_size": int(arg.batch),
                "gmm": arg.gmm
            }
        }

        # Do we want bounding boxes
        if arg.bbox is not None:
            config["wildcat_upsample"]["bounding_box"] = True
            config["wildcat_upsample"]["bounding_box_min_size"] = arg.bbox_min_size if arg.bbox_min_size > 0 else 112

        hist_train = []
        hist_val = []

    # Transforms for training and validation
    input_size = config['wildcat_upsample']['input_size']
    data_dir = os.path.join(exp_dir, "patches")

    # Create a list of transforms to compose
    data_transform_lists = {
        'train': [
            util.NormalizeRGB([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ],
        'val': [
            util.NormalizeRGB([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.CenterCrop(input_size)
        ],
    }

    # Select the type of crop operation (random or centered)
    if bool(arg.random_crop):
        print('Augmentation using random crop transform')
        data_transform_lists['train'].append(util.RandomCrop(input_size))    
    else:
        print('Augmentation using random rotation and center crop transform')
        data_transform_lists['train'].append(transforms.RandomRotation(45))    
        data_transform_lists['train'].append(transforms.CenterCrop())    

    # Append the flip augmentations
    data_transform_lists['train'].append(transforms.RandomVerticalFlip())
    data_transform_lists['train'].append(transforms.RandomHorizontalFlip())

    # Add the optional erasing transform
    if bool(arg.erasing):
        print('Augmentation using random erasing transform')
        data_transform_lists['train'].append(transforms.RandomErasing())

    if arg.color_jitter is not None:
        print('Augmentation using color jitter transform:', arg.color_jitter)
        data_transform_lists['train'].insert(0, util.ColorJitterRGB(
            arg.color_jitter[0], arg.color_jitter[1], arg.color_jitter[2], arg.color_jitter[3]))

    # Create image datasets
    data_transforms = {k: transforms.Compose(v) for k, v in data_transform_lists.items()}
    if config["wildcat_upsample"]["bounding_box"]:
        bbox_manifest = pd.read_csv(arg.bbox) 
        bbox_max = config['wildcat_upsample']['bounding_box_min_size']
    else:
        bbox_manifest = None
        bbox_max = 0
    image_datasets = {k: util.ImageFolderWithBBox(os.path.join(data_dir, k), bbox_manifest, v, bbox_max) for k, v in data_transforms.items()}

    # Get the class name to index mapping
    config['class_to_idx'] = image_datasets['train'].class_to_idx
    config['num_classes'] = len(image_datasets['train'].class_to_idx)

    # Print the config
    print("Training configuration:")
    print(config)

    # Instantiate WildCat model, loss and optiizer
    model, criterion, optimizer = make_model(config)

    # Load the model if resuming
    if bool(arg.resume) is True:
        model.load_state_dict(torch.load(model_file))

    # Wrap the model in a bounding box model
    model_bb = models.BoundingBoxWildCatModel(model)

    # Generate a weighted sampler
    class_counts = np.array(list(map(lambda x: image_datasets['train'].targets.count(x), range(config['num_classes']))))
    sample_weights = torch.DoubleTensor([1. / class_counts[i] for i in image_datasets['train'].targets])
    weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights),
                                                                      replacement=True)

    # Training and validation data loaders
    dataloaders_dict = {
        'train': torch.utils.data.DataLoader(image_datasets['train'],
                                             batch_size=config['wildcat_upsample']['batch_size'],
                                             sampler=weighted_sampler,
                                             num_workers=4),
        'val': torch.utils.data.DataLoader(image_datasets['val'],
                                           batch_size=config['wildcat_upsample']['batch_size'],
                                           shuffle=True,
                                           num_workers=4),
    }

    # Map stuff to the device
    model_bb = model_bb.to(device)
    criterion = criterion.to(device)

    # Run the training
    model_ft, hist_val_run, hist_train_run = \
        train_model(model_bb, dataloaders_dict, criterion, optimizer,
                    num_epochs=config['wildcat_upsample']['num_epochs'])

    # Append the histories
    hist_val.append(hist_val_run)
    hist_train.append(hist_train_run)

    # Save the model
    torch.save(model_ft.wildcat_model.state_dict(), model_file)

    # Add training history to saved config
    config['train_history'] = {
        'train': hist_train,
        'val': hist_val
    }

    # Save the configuration
    with open(config_file, 'w') as jfile:
        json.dump(config, jfile)


# Function to show a batch of images from Pytorch
def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


# Function to plot false positives or negatives
def plot_error(img, j, err_type, class_names, cm):
    num_fp = img[j].shape[0]
    sub_fp = np.random.choice(num_fp, min(14, num_fp), replace=False)
    if num_fp > 0:
        plt.figure(figsize=(16, 16))
        show(torchvision.utils.make_grid(img[j][sub_fp, :, :, :], padding=10, nrow=7, normalize=True))
    else:
        plt.figure(figsize=(16, 2))
    marginal = cm[j, :] if err_type == 'positives' else cm[:, j]
    plt.title("Examples of false %s for %s: (%d out of %d patches)" %
              (err_type, class_names[j], sum(marginal) - marginal[j], sum(marginal)))


def do_val(arg):
    global device

    # Read the experiment directory
    exp_dir = arg.expdir

    # Set main directories
    data_dir = os.path.join(exp_dir, "patches")
    model_dir = os.path.join(exp_dir, "models")

    # Load model configuration from config.json
    with open(os.path.join(model_dir, 'config.json'), 'r') as jfile:
        config = json.load(jfile)

    # Set global properties
    num_classes = config['num_classes']
    input_size = config['wildcat_upsample']['input_size']

    # Batch size should be input
    batch_size = arg.batch

    # Create the model
    model_ft, _, _ = make_model(config)

    # Read model state
    model_ft.load_state_dict(
        torch.load(os.path.join(model_dir, "wildcat_upsample.dat")))

    # Wrap the model as a bounding box model
    model_bb = models.BoundingBoxWildCatModel(model_ft)

    # Send model to GPU
    model_bb = model_bb.to(device)
    model_bb.eval()

    # Create a data loader
    dt = transforms.Compose([
        transforms.CenterCrop(input_size),
        util.NormalizeRGB([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the manifest if using bounding boxes
    if config["wildcat_upsample"]["bounding_box"]:
        bbox_manifest = pd.read_csv(arg.bbox) 
        bbox_max = config['wildcat_upsample']['bounding_box_min_size']
    else:
        bbox_manifest = None
        bbox_max = 0
    
    ds = util.ImageFolderWithBBox(os.path.join(data_dir, "test"), bbox_manifest, dt, bbox_max)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

    # Create array of class names
    class_names = [''] * num_classes
    for k, v in config['class_to_idx'].items():
        class_names[v] = k

    # Perform full test set evaluation and save examples of errors
    cm = np.zeros((num_classes, num_classes))
    img_fp = [torch.empty(0)] * num_classes
    img_fn = [torch.empty(0)] * num_classes

    with torch.no_grad():
        for mb, (img, label) in enumerate(dl):
            # Pass the images through the model
            outputs = model_bb(img.to(device)).cpu()

            # Get the predictions
            _, preds = torch.max(outputs, 1)

            # Print minimatch stats
            print('MB %04d/%04d  corr %d' %
                  (mb, len(dl), torch.sum(preds == label.data).item()))

            for a in range(0, len(label)):
                l_pred = preds[a].item()
                l_true = label[a].item()
                cm[l_pred, l_true] = cm[l_pred, l_true] + 1

                # Keep track of false positives and false negatives for each class
                if l_pred != l_true:
                    img_a = img[a:a + 1, :, :, :]
                    img_fp[l_pred] = torch.cat((img_fp[l_pred], img_a))
                    img_fn[l_true] = torch.cat((img_fn[l_true], img_a))

    # Path for the report
    report_dir = os.path.join(exp_dir, 'test_model')
    os.makedirs(report_dir, exist_ok=True)

    # JSon for the report
    report = {
        'confusion_matrix': cm.tolist(),
        'overall_accuracy': np.sum(np.diag(cm)) / np.sum(cm),
        'class_accuracy': {k: 1 - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - 2 * cm[i, i]) / np.sum(cm)
                           for k, i in config['class_to_idx'].items()}
    }

    # Generate true and false positives
    for j in range(num_classes):
        plot_error(img_fp, j, 'positives', class_names, cm)
        plt.savefig(os.path.join(report_dir, "false_positives_%s.png" % (class_names[j],)))

    for j in range(num_classes):
        plot_error(img_fn, j, 'negatives', class_names, cm)
        plt.savefig(os.path.join(report_dir, "false_negatives_%s.png" % (class_names[j],)))

    # Save the report
    with open(os.path.join(report_dir, 'stats.json'), 'w') as jfile:
        json.dump(report, jfile)

    # Visualize heat maps

    # Do a manual forward run of the model
    if type(model_ft) == UNet_WSL_GMM:

        # Read a batch of data
        for j in range(10):
            img, label = next(iter(dl))
            img = img[:,0:3,:,:]
            img_d = img.to(device)

            with torch.no_grad():
                x_clas = model_ft.forward_to_classifier(img_d)
                x_cpool = torch.softmax((x_clas.permute(0, 2, 3, 1) @ model_ft.fc_pooled.weight.permute(1, 0)
                                         + model_ft.fc_pooled.bias.view(1, 1, 1, -1)).permute(0, 3, 1, 2), 1)

            nnb, nnk, _, _ = x_cpool.shape
            plt.figure(figsize=((1 + nnk) * 2, nnb * 2))
            for b in range(nnb):
                plt.subplot(nnb, (1 + nnk), (1 + nnk) * b + 1)
                plt.imshow(img[b, :, :, :].permute(1, 2, 0).detach().cpu().numpy() * 0.3)
                for k in range(nnk):
                    plt.subplot(nnb, (1 + nnk), (1 + nnk) * b + k + 2)
                    plt.imshow(x_cpool[b, k, :, :].squeeze().detach().cpu().numpy(),
                               vmin=1.0/nnk, vmax=2.0/nnk)

            plt.savefig(os.path.join(report_dir, "example_activations_%02d.png" % j))

    else:
        # Read a batch of data
        img, label = next(iter(dl))
        img = img[:,0:3,:,:]
        img_d = img.to(device)
        plt.figure(figsize=(16, 16))
        show(torchvision.utils.make_grid(img, padding=10, nrow=8, normalize=True))
        plt.savefig(os.path.join(report_dir, "example_patches.png"))

        x_clas = model_ft.forward_to_classifier(img_d)
        x_cpool = model_ft.spatial_pooling.class_wise(x_clas)

        # Generate burden maps for each class
        for mode in ('activation', 'softmax'):
            for j in range(num_classes):
                plt.figure(figsize=(20, 5))
                plt.suptitle('Class %s %s' % (class_names[j], mode), fontsize=14)
                for i in range(0, batch_size):
                    plt.subplot(2, batch_size // 2, i + 1)
                    plt.title(class_names[label[i]])
                    activation = x_cpool[i, :, :, :].cpu().detach().numpy()
                    if mode == 'softmax':
                        softmax = scipy.special.softmax(activation, axis=0)
                        plt.imshow(softmax[j, :, :], vmin=0, vmax=1, cmap=plt.get_cmap('jet'))
                    else:
                        plt.imshow(activation[j, :, :], vmin=0, vmax=12, cmap=plt.get_cmap('jet'))
                plt.savefig(os.path.join(report_dir, "example_%s_%s.png" % (class_names[j], mode)))


# Set up an argument parser
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

# Configure the training parser
train_parser = subparsers.add_parser('train')
train_parser.add_argument('--expdir', help='experiment directory')
train_parser.add_argument('--epochs', help='number of epochs', default=50)
train_parser.add_argument('--batch', help='batch size', default=16)
train_parser.add_argument('--kmax', help='WildCat k_max parameter', default=0.02)
train_parser.add_argument('--kmin', help='WildCat k_min parameter', default=0.0)
train_parser.add_argument('--alpha', help='WildCat alpha parameter', default=0.7)
train_parser.add_argument('--nmaps', help='WildCat number of maps parameter', default=4)
train_parser.add_argument('--bbox', metavar='manifest.csv',
                          help='Limit loss computation to user-drawn bounding boxes, specified in the manifest file')
train_parser.add_argument('--bbox-min-size', type=int, default=0,
                          help='Minimum bounding box size, default is 112')
train_parser.add_argument('--random-crop', action='store_true',
                          help='Randomly crop input patch instead of center cropping. Recommended with bbox')  
train_parser.add_argument('--color-jitter', type=float, nargs=4, default=None,
                          help='Whether to perform color jitter augmentation (see ColorJitter in torch)')                          
train_parser.add_argument('--erasing', action='store_true',
                          help='Whether to perform erasing augmentation')
train_parser.add_argument('--gmm', type=int, metavar='N', default=0,
                          help='Use Gaussian Mixture Model network with N iterations of EM')
train_parser.add_argument('--resume', action='store_true',
                          help='Resume training using previously saved parameters')
train_parser.set_defaults(func=do_train)

# Configure the validation parser
val_parser = subparsers.add_parser('validate')
val_parser.add_argument('--expdir', help='experiment directory')
val_parser.add_argument('--bbox', metavar='manifest.csv',
                        help='Limit loss computation to user-drawn bounding boxes, specified in the manifest file')
val_parser.add_argument('--bbox-min-size', type=int, default=0,
                        help='Minimum bounding box size, default is 112')
val_parser.add_argument('--batch', help='batch size', default=16)                        
val_parser.set_defaults(func=do_val)

# Configure the apply parser
apply_parser = subparsers.add_parser('apply')
apply_parser.add_argument('--slide', help='Input histology slide to process')
apply_parser.add_argument('--modeldir', help='Directory containing the model')
apply_parser.add_argument('--output', help='Where to store the output density map')
apply_parser.add_argument('--region', help='Region of image to process (x,y,w,h)', nargs=4)
apply_parser.add_argument('--window', help='Size of the window for scanning', default=4096)
apply_parser.add_argument('--shrink', help='How much to downsample WildCat output', default=4)
apply_parser.set_defaults(func=do_apply)

# Configure the patch apply parser
patch_apply_parser = subparsers.add_parser('patch_apply')
patch_apply_parser.add_argument('--input', help='Input directory containing patches to process')
patch_apply_parser.add_argument('--modeldir', help='Directory containing the model')
patch_apply_parser.add_argument('--output', help='Output file where to store density values')
patch_apply_parser.add_argument('--outdir', help='Output directory where to store density maps (optional)')
patch_apply_parser.add_argument('--outstat', help='Output file where to store density histograms (optional)')
patch_apply_parser.add_argument('--shrink', help='How much to downsample WildCat output', default=4)
patch_apply_parser.add_argument('--batch', help='Batch size', default=16, type=int)
patch_apply_parser.add_argument('--scale',
                                help='When applying to images of different resolution than '
                                     'the training set, set this parameter to the ratio '
                                     'test_pixel_size / train_pixel_size', default=1.0, type=float)
patch_apply_parser.set_defaults(func=do_patch_apply)

# Configure the info parser
info_parser = subparsers.add_parser('info')
info_parser.set_defaults(func=do_info)

# Add common options to all subparsers
if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)
