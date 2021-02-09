# Set up
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
import argparse
import SimpleITK as sitk
from PIL import Image
import sys

sys.path.append("deepcluster")
from deepcluster.util import load_model


# Process a slide
def apply_to_slide(args):

    # Read the trained model
    model = load_model(args.network)
    model.cuda()
    model.eval()

    # Read the slide
    I = Image.open(args.slide)
    h,w=I.size

    # Split the transforms
    tran_tt = transforms.ToTensor()
    tran_model=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])])
    
    # Get the patch size and downsample factor
    patch_size, ds = int(args.patch), int(args.downsample)

    # Image as tensor
    I_crop=I.crop((0, 0, ds*int(h/ds), ds*int(w/ds)))
    IT = tran_tt(I_crop)

    # Image unfolded into patches
    IT_unfold = IT.unfold(1, patch_size, ds).unfold(2, patch_size, ds).permute(1, 2, 0, 3, 4)

    # Output dimensions (before padding)
    (ow, oh) = IT_unfold.shape[0], IT_unfold.shape[1]

    # Flat array of inputs
    batches = IT_unfold.reshape(-1, 3, patch_size, patch_size)

    # Break into digestable batches
    bs = int(args.batch_size)
    for k in range(0, batches.shape[0], bs):
        k_end = min(k + bs, batches.shape[0])
        batch=torch.zeros((k_end-k,3,224,224))
        for j in range(k,k_end):
            batch[j-k,:,:,:]=tran_model(batches[j,:,:,:])
        with torch.no_grad():
            res_batch=model(batch.cuda()).detach().cpu()
            if k == 0:
                result = res_batch
            else:
                result = torch.cat((result, res_batch), 0)
        print('Batch %d of %d' % (k / bs, batches.shape[0] / bs))

    # Reformat into a 20xWxH image
    Z = result.permute(1, 0).reshape(-1, ow, oh)

    # Pad the result to desired size
    owp, ohp = int(w * 1.0 / ds + 0.5), int(h * 1.0 / ds + 0.5)
    pw0, ph0 = int((owp - ow) / 2), int((ohp - oh) / 2)
    pw1, ph1 = owp - ow - pw0, ohp - oh - ph0
    Z = torch.nn.functional.pad(Z, (ph0, ph1, pw0, pw1, 0, 0), 'constant', 0)

    # Write the result as a NIFTI file
    nii = sitk.GetImageFromArray(Z.permute(1,2,0), True)
    nii.SetSpacing((ds, ds))
    sitk.WriteImage(nii, args.output)

    # Write the optional thumb
    if args.thumb is not None:
        rgb = np.asarray(I.resize((ohp,owp),Image.LANCZOS))
        nii = sitk.GetImageFromArray(rgb, True)
        nii.SetSpacing((ds, ds))
        sitk.WriteImage(nii, args.thumb)


# Set up an argument parser
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

apply_parser = subparsers.add_parser('apply')
apply_parser.add_argument('--slide', help='Input histology slide to process')
apply_parser.add_argument('--output', help='Where to store the output density map')
apply_parser.add_argument('--thumb', help='Optional thumbnail output', default=None)
apply_parser.add_argument('--network', help='Network saved during training')
apply_parser.add_argument('--patch', help='Patch size used during training', default=64)
apply_parser.add_argument('--downsample', help='Input to output downsampling ratio', default=16)
apply_parser.add_argument('--batch-size', help='GPU batch size', default=256)
apply_parser.set_defaults(func=apply_to_slide)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)
