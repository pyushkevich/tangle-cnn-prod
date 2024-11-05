#!/usr/bin/env python3
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
from .osl_worker import osl_worker, osl_read_chunk_from_queue
from .osl_worker import HistologyDataSource, OpenSlideHistologyDataSource, SimpleITKHistologyDataSource
from .wildcat_pytorch.picsl_wildcat import models, util, losses
from .unet_wildcat_gmm import UNet_WSL_GMM
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


def make_model(config, pretrained=True):
    """Instantiate a model, loss, and optimizer based on the config dict"""
    mcfg = config['wildcat_upsample']

    # Instantiate WildCat model, loss and optiizer
    if 'gmm' in mcfg and mcfg['gmm'] > 0:
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
        lr = mcfg.get('lr', 0.001)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)

    else:
        model = models.resnet50_wildcat_upsample(
            config['num_classes'],
            pretrained=True,
            kmax=mcfg['kmax'],
            kmin=mcfg['kmin'],
            alpha=mcfg['alpha'],
            num_maps=mcfg['num_maps'])

        # Loss and optimizer
        if 'mlloss_weights' in mcfg:
            w = torch.tensor(mcfg['mlloss_weights'], device=device)
            criterion = losses.TanglethonLoss(w)
        else:
            criterion = losses.MultiLabelSoftMarginLoss()

        # Initialize the optimizer
        lr = mcfg.get('lr', 0.01)
        optimizer = torch.optim.SGD(model.get_config_optim(lr, 0.1), lr=lr, momentum=0.9, weight_decay=1e-2)

    return model, criterion, optimizer

class TrainedWildcat:
    
    def __init__(self, modeldir, device=None):
        
        # Set the model directory
        self.model_dir = modeldir
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        print(f'TrainedWildcat {modeldir} on device {self.device}')

        # Read the model's .json config file
        with open(os.path.join(self.model_dir, 'config.json')) as json_file:
            self.config = json.load(json_file)
        
        # Create the model
        self.model_ft, _, _ = make_model(self.config, pretrained=False)

        # Read model state
        self.model_ft.load_state_dict(
            torch.load(os.path.join(self.model_dir, "wildcat_upsample.dat"),
                       weights_only=True, map_location=self.device))

        # Send model to GPU
        self.model_ft.eval()
        self.model_ft = self.model_ft.to(self.device)
        
    @property
    def suffix(self):
        return self.config['suffix']
    
    def apply(self, osl:HistologyDataSource, 
              window_size:int, extra_shrinkage:int=0, 
              region=None, target_resolution=None, crop=False):
        """Apply Wildcat to a histology slide.
        
        Applies the trained Wildcat model to sliding windows passed over the histology
        slide and generates a multi-component Nitfi image with Wildcat activation maps.
        
        Args:
            osl (HistologyDataSource): Object from which histology will be sampled. Can be
                ``OpenSlideHistologyDataSource`` or ``SimpleITKHistologyDataSource``
            window_size (int): Size of the window used to apply WildCat. 
                Should be larger than the patch size used for training.
            extra_shrinkage (int, optional): Additional shrinkage factor to apply to output 
                (because we don't want to store very large outputs)
            region (list of float or int, optional): Restrict the extraction to a region, specified in the
                format [x,y,w,h]. If all four numbers are in range [0, 1], the region specification is
                relative to the image size. If not, the region specification is in units of windows.
            target_resolution (float, optional): Target spacing to which the input data should be 
                resampled before performing inference. This should be used when the spacing 
                (resolution) of the slide is different from the resolution on which the model
                was trained.
            crop (bool, optional): If True, the output image will be cropped to the region processed. 
                Otherwise, the density image corresponding to the whole slide is returned. Has no effect
                if region is None. Default: False.
        Returns:
            Density image as `SimpleITK.Image`. The density image is a multichannel 2D image in
            which every pixel stores the posterior probability of each tissue class.
        """
        
        # Read the input using OpenSlide
        slide_dim = np.array(osl.dimensions)

        # Input size to WildCat (should be 224)
        input_size_wildcat = self.config['wildcat_upsample']['input_size']
        
        # Corresponding patch size in the histology image
        res_factor = 1 if target_resolution is None else target_resolution / osl.spacing[0]
        patch_size_raw = int(0.5 + input_size_wildcat * res_factor)

        # Corresponding patch size in raw image (should match WildCat, no downsampling)
        window_size_raw = int(0.5 + window_size * res_factor)

        # The amount of padding, relative to patch size to add to the window. This padding
        # is to provide context at the edges of the window
        padding_size_rel = 1.0
        padding_size_raw = int(padding_size_rel * patch_size_raw)

        # Factor by which wildcat shrinks input images when mapping to segmentations
        wildcat_shrinkage = 2

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
        num_classes = self.config['num_classes']
        density = np.zeros((num_classes, out_dim[0], out_dim[1]))

        # Range of pixels to scan
        u_range, v_range = (0, n_win[0]), (0, n_win[1])

        # Allow a custom region to be specified
        if region is not None and len(region) == 4:
            region = list(float(val) for val in region)
            if all(val <= 1.0 for val in region):
                u_range, v_range = (
                    (slide_dim[i] * region[i] // window_size_raw,
                     1 + slide_dim[i] * (region[i] + region[i+2]) // window_size_raw) for i in range(2))
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
                    chunk_tensor = torch.unsqueeze(tran(chunk_img), dim=0).to(self.device)

                    # Forward pass through the wildcat model
                    x_clas = self.model_ft.forward_to_classifier(chunk_tensor)
                    x_cpool = self.model_ft.spatial_pooling.class_wise(x_clas)

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

            # Report total time
            t_11 = timeit.default_timer()
            print("Total time elapsed: %8.4f" % (t_11 - t_00,))

        except:
            traceback.print_exc()
            sys.exit(-1)

        finally:
            worker.join(60)
            if worker.is_alive():
                print('Thread worker failed to terminate after 60 seconds')

        # Crop the density image either to the whole image size or to the size of the
        # region processed
        out_dim_trim = np.round((slide_dim / out_pix_size)).astype(int)
        d_x0 = u_range[0] * window_size_out if crop else 0
        d_x1 = min(out_dim_trim[0], u_range[1] * window_size_out) if crop else out_dim_trim[0]
        d_y0 = v_range[0] * window_size_out if crop else 0
        d_y1 = min(out_dim_trim[1], v_range[1] * window_size_out) if crop else out_dim_trim[1]

        # Extract the desired region of the density image and convert to ITK image
        density = density[:, d_x0:d_x1, d_y0:d_y1]
        nii = sitk.GetImageFromArray(np.transpose(density, (2, 1, 0)), isVector=True)
        
        # Set the spacing based on openslide        
        sx, sy = (osl.spacing * out_pix_size).tolist()
        nii.SetSpacing((sx, sy))
        nii.SetOrigin(((d_x0+0.5) * sx, (d_y0+0.5) * sy))

        return nii
        
    # Function to apply training to a collection of extracted patches
    def apply_to_patches(self, fn_input, fn_outdir, fn_outstat, scale, batch_size):

        # Number of classes in the model
        num_classes = self.config['num_classes']

        # Create a transform
        tran = transforms.Compose([
            ResizeByFactor(scale),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Create a data set and data loader
        ds = SimplePNGFolder(fn_input, tran)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=4)

        # Find all the input files
        allstat = {}
    
        # Process data in batches
        for ib, (img, paths) in enumerate(dl):
            print('Directory {} Batch {} of {}'.format(fn_input, ib, len(dl)))

            # Process minibatch through model
            with torch.no_grad():
                x_clas = self.model_ft.forward_to_classifier(img.to(self.device))
                x_cpool = self.model_ft.spatial_pooling.class_wise(x_clas)

            z = x_cpool.cpu().detach().numpy()

            # Collect statistics from the minibatch
            for j in range(img.shape[0]):
                zstat = []
                for k in range(num_classes):
                    zk = x_cpool[j,k,:,:].detach().cpu().numpy().flatten()
                    zhist = np.histogram(zk, bins=np.linspace(-6.0, 6.0, 121), density=True)
                    zmean = np.mean(zk)
                    zstd = np.std(zk)
                    zstat.append({'histogram': zhist[0].tolist(), 'mean': zmean.item(), 'std': zstd.item()})
                
                # Append the statistics
                allstat[os.path.basename(paths[j])] = zstat

                # Write the image back to destination dir if requested
                if fn_outdir:
                    fn_dest = os.path.join(fn_outdir, "wildcat_" + os.path.splitext(os.path.basename(paths[j]))[0] + ".nii.gz")
                    dimg = x_cpool[j,:,:,:].cpu().numpy().transpose(1,2,0)
                    nii = sitk.GetImageFromArray(dimg, True)
                    sitk.WriteImage(nii, fn_dest)

        # Write the output JSON file, if requested
        if fn_outstat:
            with open(fn_outstat, "w") as f_outstat:
                json.dump(allstat, f_outstat)


# Dataset for loading all images from a single directory
class SimplePNGFolder(datasets.VisionDataset):

    def __init__(self, root, transform):
        super().__init__(root=root, transform=transform)
        self.files = glob.glob(os.path.join(root, '*.png'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.files[index]


# Resize transform that takes a scaling factor
class ResizeByFactor(torch.nn.Module):

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, patch):
        size = int(patch.size[0] * self.factor),int(patch.size[1] * self.factor)
        return torchvision.transforms.functional.resize(
            patch, size, torchvision.transforms.InterpolationMode.BILINEAR)


# Function to apply training to a collection of extracted patches
def do_patch_apply(args):

    # Create a wildcat instance    
    wc = TrainedWildcat(args.modeldir)

    # Make sure the argument sizes match
    file_args = list(zip(
        args.input,
        args.outstat if args.outstat is not None else [None] * len(args.input),
        args.outdir if args.outdir is not None else [None] * len(args.input),
        args.scale if args.scale is not None else [1.0] * len(args.input)))

    # Iterate over input tuples
    for (input, outstat, outdir, scale) in file_args:
        wc.apply_to_patches(input, outdir, outstat, scale, args.batch)


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
            for mb, (inputs, labels, patch_ids) in enumerate(dataloaders[phase]):
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
                    preds = criterion.predictions(outputs)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()

                        for param in optimizer.param_groups[0]['params']:
                            if param.grad is not None:
                                valid_gradients = not (torch.isnan(param.grad).any())
                                if not valid_gradients:
                                    break

                        if not valid_gradients:
                            print("detected inf or nan values in gradients. not updating model parameters")
                            optimizer.zero_grad()
                        else:
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
                "gmm": arg.gmm,
                "lr": arg.lr
            }
        }

        # Check if the manifest is provided
        if arg.bbox is True or arg.scale is not None:
            if arg.manifest is None:
                raise ValueError('Missing manifest parameter for bbox/scale options') 

        # Do we want bounding boxes
        if arg.bbox is True:
            config["wildcat_upsample"]["bounding_box"] = True
            config["wildcat_upsample"]["bounding_box_min_size"] = arg.bbox_min_size if arg.bbox_min_size > 0 else 112

        # Do we want scaling to target resolution
        if arg.scale is not None:
            config["wildcat_upsample"]["target_resolution"] = arg.scale

        # Do we want to use a multi-label loss
        mlloss_spec=None
        if arg.mlloss is not None:
            mlloss_spec = json.load(arg.mlloss)
            config["wildcat_upsample"]["mlloss"] = mlloss_spec

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

    # Read the manifest
    manifest = pd.read_csv(arg.manifest) if arg.manifest is not None else None

    # Are we using bounding boxes?
    use_bbox = config["wildcat_upsample"].get('bounding_box', False)
    bbox_min = config['wildcat_upsample'].get('bounding_box_min_size', 0)
    target_resolution = config["wildcat_upsample"].get('target_resolution', None)

    image_datasets = {k: util.ImageFolderWithBBox(
        os.path.join(data_dir, k), use_bbox, manifest, v, bbox_min, target_resolution) 
        for k, v in data_transforms.items()}

    # Get the class name to index mapping
    config['class_to_idx'] = image_datasets['train'].class_to_idx
    config['num_classes'] = len(image_datasets['train'].class_to_idx)

    # Generate the matrix of weights for the TanglethonLoss
    if mlloss_spec is not None:
        mlloss_mat = np.zeros((config['num_classes'],config['num_classes']))
        for (k1,v1) in config['class_to_idx'].items():
            for (k2, v2) in config['class_to_idx'].items():
                mlloss_mat[v1,v2] = mlloss_spec.get('{},{}'.format(k1,k2), 0.0)
        config["wildcat_upsample"]["mlloss_weights"] = mlloss_mat.tolist()

    # Print the config
    print("Training configuration:")
    print(config)

    # Instantiate WildCat model, loss and optiizer
    model, criterion, optimizer = make_model(config, pretrained=True)

    # Load the model if resuming
    if bool(arg.resume) is True:
        model.load_state_dict(torch.load(model_file, weights_only=True))

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
def plot_error_old(img, j, err_type, class_names, cm):
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


def plot_error(d_err, j, err_type, class_names, cm):
    num_err = len(d_err[j])
    sub_err = np.random.choice(num_err, min(16, num_err), replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(16,16))
    for p, ax in enumerate(axes.flat):
        if p < len(sub_err):
            i = sub_err[p]
            patch = d_err[j][i]['patch']
            nm_true = class_names[d_err[j][i]['true']]
            nm_pred = class_names[d_err[j][i]['pred']]
            ax.imshow((patch[:,:,0:3] + 2.2) / 5)
            if patch.shape[2] > 3:
                ax.imshow(patch[:,:,-1], alpha = 0.2, vmin=0, vmax=1)
            ax.set_axis_off()
            ax.set_title(f'Y="{nm_pred}", T="{nm_true}"')
        else:
            ax.set_axis_off()

    # Report statistics
    (marginal, nm_rate) = (cm[j, :],'FPR') if err_type == 'positives' else (cm[:, j], 'FNR')
    s_marginal = sum(marginal)
    n_err = sum(marginal) - marginal[j]
    err_rate = n_err / s_marginal
    fig.suptitle('Examples of false %s for class "%s". %s = %6.4f (%d / %d)' %
                 (err_type, class_names[j], nm_rate, err_rate, n_err, s_marginal))
    return fig


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
    model_ft, criterion, _ = make_model(config, pretrained=False)

    # Read model state
    model_ft.load_state_dict(
        torch.load(os.path.join(model_dir, "wildcat_upsample.dat"), weights_only=True))

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

    # Read the manifest
    manifest = pd.read_csv(arg.manifest) if arg.manifest is not None else None

    # Load the manifest if using bounding boxes
    use_bbox = config["wildcat_upsample"].get('bounding_box', False)
    bbox_min = config['wildcat_upsample'].get('bounding_box_min_size', 0)
    target_resolution = config["wildcat_upsample"].get('target_resolution', None)
    
    ds = util.ImageFolderWithBBox(
        os.path.join(data_dir, arg.target), use_bbox, manifest, dt, bbox_min, target_resolution)

    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

    # Create array of class names
    class_names = [''] * num_classes
    for k, v in config['class_to_idx'].items():
        class_names[v] = k

    # Perform full test set evaluation and save examples of errors
    cm = np.zeros((num_classes, num_classes))
    
    # False positive and false negative arrays are lists of dicts, each
    # dict stores an image patch, mask if available, patch id, etc.
    d_fp = list([ [] for k in range(num_classes) ])
    d_fn = list([ [] for k in range(num_classes) ])
    
    # Keep track of what patch was assigned what class
    all_patch_ids, all_patch_pred, all_patch_true = [], [], []

    with torch.no_grad():
        for mb, (img, label, patch_ids) in enumerate(dl):
            # Pass the images through the model
            outputs = model_bb(img.to(device))

            # Get the predictions
            preds = criterion.predictions(outputs).cpu()

            # Print minimatch stats
            print('MB %04d/%04d  corr %d' %
                  (mb, len(dl), torch.sum(preds == label.data).item()))

            for a in range(0, len(label)):
                l_pred = preds[a].item()
                l_true = label[a].item()
                cm[l_pred, l_true] = cm[l_pred, l_true] + 1

                # Append predictions to list of all predictions
                all_patch_ids.append(patch_ids[a])
                all_patch_pred.append(l_pred)
                all_patch_true.append(l_true) 

                # Keep track of false positives and false negatives for each class
                if l_pred != l_true:
                    err = {
                        'id': patch_ids[a],
                        'patch': img[a, :, :, :].permute(1,2,0).detach().cpu().numpy(),
                        'true': l_true,
                        'pred': l_pred
                    }
                    d_fp[l_pred].append(err)
                    d_fn[l_true].append(err)

    # Path for the report
    fld_name = 'test_model' if arg.target == 'test' else f'test_model_{arg.target}'
    report_dir = os.path.join(exp_dir, fld_name)
    os.makedirs(report_dir, exist_ok=True)

    # JSon for the report
    report = {
        'confusion_matrix': cm.tolist(),
        'overall_accuracy': np.sum(np.diag(cm)) / np.sum(cm),
        'class_accuracy': {k: 1 - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - 2 * cm[i, i]) / np.sum(cm)
                           for k, i in config['class_to_idx'].items()}
    }

    # Write individual patch predictions if requested
    df_patch = pd.DataFrame({'sample': all_patch_ids, 'prediction': all_patch_pred, 'label': all_patch_true})
    df_patch.to_csv(os.path.join(report_dir, 'patch_predictions.csv'), index=False)

    # Generate true and false positives
    for j in range(num_classes):
        fig = plot_error(d_fp, j, 'positives', class_names, cm)
        fig.savefig(os.path.join(report_dir, "false_positives_%s.png" % (class_names[j],)))
        plt.close(fig)

    for j in range(num_classes):
        fig = plot_error(d_fn, j, 'negatives', class_names, cm)
        fig.savefig(os.path.join(report_dir, "false_negatives_%s.png" % (class_names[j],)))
        plt.close(fig)

    # Save the report
    with open(os.path.join(report_dir, 'stats.json'), 'w') as jfile:
        json.dump(report, jfile)

    # Visualize heat maps

    # Do a manual forward run of the model
    if type(model_ft) == UNet_WSL_GMM:

        # Read a batch of data
        for j in range(10):
            img, label, patch_id = next(iter(dl))
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
        img_msk, label, patch_id = next(iter(dl))
        img = img_msk[:,0:3,:,:]
        img_d = img.to(device)
        plt.figure(figsize=(16, 16))
        util.show_patches_with_masks(img_msk, labels=label, class_names=class_names)
        # show(torchvision.utils.make_grid(img, padding=10, nrow=8, normalize=True))
        plt.savefig(os.path.join(report_dir, "example_patches.png"))

        with torch.no_grad():
            x_clas = model_ft.forward_to_classifier(img_d)
            x_cpool = model_ft.spatial_pooling.class_wise(x_clas)

        # Generate burden maps for each class
        for mode in ('activation', 'softmax'):
            for j in range(num_classes):
                plt.figure(figsize=(20, 5))
                plt.suptitle('Class %s %s' % (class_names[j], mode), fontsize=14)
                for i in range(0, batch_size):
                    plt.subplot(2, (batch_size+1) // 2, i + 1)
                    plt.title(class_names[label[i]])
                    activation = x_cpool[i, :, :, :].cpu().detach().numpy()
                    if mode == 'softmax':
                        softmax = scipy.special.softmax(activation, axis=0)
                        plt.imshow(softmax[j, :, :], vmin=0, vmax=1, cmap=plt.get_cmap('jet'))
                    else:
                        plt.imshow(activation[j, :, :], vmin=0, vmax=12, cmap=plt.get_cmap('jet'))
                plt.savefig(os.path.join(report_dir, "example_%s_%s.png" % (class_names[j], mode)))


# Function to apply training to a slide
def do_apply(args):
    wc = TrainedWildcat(args.modeldir)
    if args.reader == 'osl':
        osl = openslide.OpenSlide(args.slide)
        datasource = OpenSlideHistologyDataSource(args.slide)
    elif args.reader == 'sitk':
        img = sitk.ReadImage(args.slide)
        datasource = SimpleITKHistologyDataSource(img)
    else:
        raise ValueError(f'Unsupported reader {args.reader}')
    
    nii = wc.apply(datasource, args.window, args.shrink, args.region)
    sitk.WriteImage(nii, args.output)

# Set up an argument parser
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

# Configure the training parser
train_parser = subparsers.add_parser('train')
train_parser.add_argument('--expdir', help='experiment directory')
train_parser.add_argument('--epochs', help='number of epochs', default=50)
train_parser.add_argument('--batch', help='batch size', default=16, type=int)
train_parser.add_argument('--kmax', help='WildCat k_max parameter', default=0.02)
train_parser.add_argument('--kmin', help='WildCat k_min parameter', default=0.0)
train_parser.add_argument('--alpha', help='WildCat alpha parameter', default=0.7)
train_parser.add_argument('--nmaps', help='WildCat number of maps parameter', default=4)
train_parser.add_argument('--manifest', help='Manifest file for --bbox/--scale options')
train_parser.add_argument('--bbox', action='store_true',
                          help='Limit loss computation to user-drawn bounding boxes, specified in the manifest file')
train_parser.add_argument('--bbox-min-size', type=int, default=0,
                          help='Minimum bounding box size, default is 112')
train_parser.add_argument('--random-crop', action='store_true',
                          help='Randomly crop input patch instead of center cropping. Recommended with bbox')  
train_parser.add_argument('--scale', metavar='target_resolution', type=float, default=None,
                          help="""Scale the patches to given physical resolution before processing, use when you have
                                  slides with different resolution, manifest must include mpp_x and mpp_y columns.
                                  Target resolution must be specified in mm per pixel""")
train_parser.add_argument('--color-jitter', type=float, nargs=4, default=None,
                          help='Whether to perform color jitter augmentation (see ColorJitter in torch)')                          
train_parser.add_argument('--erasing', action='store_true',
                          help='Whether to perform erasing augmentation')
train_parser.add_argument('--gmm', type=int, metavar='N', default=0,
                          help='Use Gaussian Mixture Model network with N iterations of EM. NOT RECOMMENDED.')
train_parser.add_argument('--resume', action='store_true',
                          help='Resume training using previously saved parameters')
train_parser.add_argument('--mlloss', type=argparse.FileType('r'), 
                          help="""Use a multi-label loss with the weights specified in the JSON file.
                                  The format of the weights should be 
                                    {
                                    "thread,tangle": 0.5,
                                    "tangle,other": 1.0,
                                    ...
                                    }
                                  where all non-zero entries in the weight matrix for the TanglethonLoss should
                                  be included. The snipped above reads, "if the true label of a patch is
                                  thread, the loss associated with labeling it as tangle is 0.5".
                                """)
train_parser.add_argument('--lr', help='Learning rate, default 0.01 (0.001 for GMM)', default=None, type=float)
train_parser.set_defaults(func=do_train)

# Configure the validation parser
val_parser = subparsers.add_parser('validate')
val_parser.add_argument('--expdir', help='experiment directory')
val_parser.add_argument('--manifest', metavar='manifest.csv',
                        help='Manifest file needed if training was performed with --bbox or --scale')
val_parser.add_argument('--batch', help='batch size', default=16, type=int)                       
val_parser.add_argument('--target', help='which data to do evaluation on', default='test')                       
val_parser.set_defaults(func=do_val)

# Configure the apply parser
apply_parser = subparsers.add_parser('apply')
apply_parser.add_argument('--slide', help='Input histology slide to process')
apply_parser.add_argument('--reader', choices=['openslide', 'pillow', 'sitk'], help='Reader to use for the slide', default='openslide')
apply_parser.add_argument('--modeldir', help='Directory containing the model')
apply_parser.add_argument('--output', help='Where to store the output density map')
apply_parser.add_argument('--region', help='Region of image to process (x,y,w,h)', nargs=4)
apply_parser.add_argument('--window', help='Size of the window for scanning', default=4096)
apply_parser.add_argument('--shrink', help='How much to downsample WildCat output', default=4)
apply_parser.add_argument('--manifest', type=str, help='Read input and output data from a manifest .csv file, columns must include slide,output')
apply_parser.set_defaults(func=do_apply)

# Configure the patch apply parser
patch_apply_parser = subparsers.add_parser('patch_apply')
patch_apply_parser.add_argument('--input', help='Input directory containing patches to process', nargs='+')
patch_apply_parser.add_argument('--modeldir', help='Directory containing the model')
patch_apply_parser.add_argument('--outdir', help='Output directory where to store density maps (optional)', nargs='+')
patch_apply_parser.add_argument('--outstat', help='Output file where to store density histograms (optional)', nargs='+')
patch_apply_parser.add_argument('--shrink', help='How much to downsample WildCat output', default=4)
patch_apply_parser.add_argument('--batch', help='Batch size', default=16, type=int)
patch_apply_parser.add_argument('--scale',
                                help='When applying to images of different resolution than '
                                     'the training set, set this parameter to the ratio '
                                     'test_pixel_size / train_pixel_size', nargs='+', type=float)
patch_apply_parser.set_defaults(func=do_patch_apply)

# Configure the info parser
info_parser = subparsers.add_parser('info')
info_parser.set_defaults(func=do_info)


# Add common options to all subparsers
#if __name__ == '__main__':
#    args = parser.parse_args()
#    args.func(args)
