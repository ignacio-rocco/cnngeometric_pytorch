from __future__ import print_function, division
import os
import argparse
from torch.utils.data import DataLoader
from model.cnn_geometric_model import CNNGeometric
from data.pf_dataset import PFDataset
from data.download_datasets import download_PF_willow
from image.normalization import NormalizeImageDict
from util.torch_util import BatchTensorToVars
from geotnf.point_tnf import *
from geotnf.transformation import GeometricTnf
from collections import OrderedDict

"""

Script to evaluate a trained model as presented in the CNNGeometric CVPR'17 paper
on the ProposalFlow dataset

"""

print('CNNGeometric PF evaluation script')

# Argument parsing
parser = argparse.ArgumentParser(description='CNNGeometric PyTorch implementation')
# Paths
parser.add_argument('--model-aff', type=str,
                    default='trained_models/best_pascal_checkpoint_adam_affine_grid_loss_resnet_random.pth.tar',
                    help='Trained affine model filename')
parser.add_argument('--model-tps', type=str,
                    default='trained_models/best_pascal_checkpoint_adam_tps_grid_loss_resnet_random.pth.tar',
                    help='Trained TPS model filename')
parser.add_argument('--feature-extraction-cnn', type=str, default='resnet101',
                    help='Feature extraction architecture: vgg/resnet101')
parser.add_argument('--pf-path', type=str, default='datasets/PF-dataset', help='Path to PF dataset')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

do_aff = not args.model_aff == ''
do_tps = not args.model_tps == ''

# Download dataset if needed
download_PF_willow('datasets/')

# Create model
print('Creating CNN model...')
if do_aff:
    model_aff = CNNGeometric(use_cuda=use_cuda, geometric_model='affine',
                             feature_extraction_cnn=args.feature_extraction_cnn)
if do_tps:
    model_tps = CNNGeometric(use_cuda=use_cuda, geometric_model='tps',
                             feature_extraction_cnn=args.feature_extraction_cnn)

# Load trained weights
print('Loading trained model weights...')
if do_aff:
    checkpoint = torch.load(args.model_aff, map_location=lambda storage, loc: storage)
    checkpoint['state_dict'] = OrderedDict(
        [(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
    model_aff.load_state_dict(checkpoint['state_dict'])
if do_tps:
    checkpoint = torch.load(args.model_tps, map_location=lambda storage, loc: storage)
    checkpoint['state_dict'] = OrderedDict(
        [(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
    model_tps.load_state_dict(checkpoint['state_dict'])

# Dataset and dataloader
dataset = PFDataset(csv_file=os.path.join(args.pf_path, 'test_pairs_pf.csv'),
                    training_image_path=args.pf_path,
                    transform=NormalizeImageDict(['source_image', 'target_image']))
if use_cuda:
    batch_size = 16
else:
    batch_size = 1

dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4)

batchTensorToVars = BatchTensorToVars(use_cuda=use_cuda)

# Instantiate point transformer
pt = PointTnf(use_cuda=use_cuda)

# Instatiate image transformers
tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=use_cuda)
affTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)


# Compute PCK
def correct_keypoints(source_points, warped_points, L_pck, alpha=0.1):
    # compute correct keypoints
    point_distance = torch.pow(torch.sum(torch.pow(source_points - warped_points, 2), 1), 0.5).squeeze(1)
    L_pck_mat = L_pck.expand_as(point_distance)
    correct_points = torch.le(point_distance, L_pck_mat * alpha)
    num_of_correct_points = torch.sum(correct_points)
    num_of_points = correct_points.numel()
    return (num_of_correct_points, num_of_points)


print('Computing PCK...')
total_correct_points_aff = 0
total_correct_points_tps = 0
total_correct_points_aff_tps = 0
total_points = 0
for i, batch in enumerate(dataloader):

    batch = batchTensorToVars(batch)

    source_im_size = batch['source_im_size']
    target_im_size = batch['target_im_size']

    source_points = batch['source_points']
    target_points = batch['target_points']

    # warp points with estimated transformations
    target_points_norm = PointsToUnitCoords(target_points, target_im_size)

    if do_aff:
        model_aff.eval()
    if do_tps:
        model_tps.eval()

    if do_aff:
        theta_aff = model_aff(batch)

        # do affine only
        warped_points_aff_norm = pt.affPointTnf(theta_aff, target_points_norm)
        warped_points_aff = PointsToPixelCoords(warped_points_aff_norm, source_im_size)
    if do_tps:
        theta_tps = model_tps(batch)

        # do tps only
        warped_points_tps_norm = pt.tpsPointTnf(theta_tps, target_points_norm)
        warped_points_tps = PointsToPixelCoords(warped_points_tps_norm, source_im_size)

    if do_aff and do_tps:
        warped_image_aff = affTnf(batch['source_image'], theta_aff.view(-1, 2, 3))
        theta_aff_tps = model_tps({'source_image': warped_image_aff, 'target_image': batch['target_image']})

        # do tps+affine
        warped_points_aff_tps_norm = pt.tpsPointTnf(theta_aff_tps, target_points_norm)
        warped_points_aff_tps_norm = pt.affPointTnf(theta_aff, warped_points_aff_tps_norm)
        warped_points_aff_tps = PointsToPixelCoords(warped_points_aff_tps_norm, source_im_size)

    L_pck = batch['L_pck'].data

    if do_aff:
        correct_points_aff, num_points = correct_keypoints(source_points.data,
                                                           warped_points_aff.data, L_pck)
        total_correct_points_aff += correct_points_aff

    if do_tps:
        correct_points_tps, num_points = correct_keypoints(source_points.data,
                                                           warped_points_tps.data, L_pck)
        total_correct_points_tps += correct_points_tps

    if do_aff and do_tps:
        correct_points_aff_tps, num_points = correct_keypoints(source_points.data,
                                                               warped_points_aff_tps.data, L_pck)
        total_correct_points_aff_tps += correct_points_aff_tps

    total_points += num_points

    print('Batch: [{}/{} ({:.0f}%)]'.format(i, len(dataloader), 100. * i / len(dataloader)))

if do_aff:
    PCK_aff = float(total_correct_points_aff) / float(total_points)
    print('PCK affine:', PCK_aff)
if do_tps:
    PCK_tps = float(total_correct_points_tps) / float(total_points)
    print('PCK tps:', PCK_tps)
if do_aff and do_tps:
    PCK_aff_tps = float(total_correct_points_aff_tps) / float(total_points)
    print('PCK affine+tps:', PCK_aff_tps)
print('Done!')
