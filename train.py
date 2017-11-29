from __future__ import print_function, division
import argparse
import os
from os.path import exists, join, basename
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model.cnn_geometric_model import CNNGeometric
from model.loss import TransformedGridLoss
from data.synth_dataset import SynthDataset
from data.download_datasets import download_pascal
from geotnf.transformation import SynthPairTnf
from image.normalization import NormalizeImageDict
from util.train_test_fn import train, test
from util.torch_util import save_checkpoint, str_to_bool

"""

Script to train the model as presented in the CNNGeometric CVPR'17 paper
using synthetically warped image pairs and strong supervision

"""

print('CNNGeometric training script')

# Argument parsing
parser = argparse.ArgumentParser(description='CNNGeometric PyTorch implementation')
# Paths
parser.add_argument('--training-dataset', type=str, default='pascal', help='dataset to use for training')
parser.add_argument('--training-tnf-csv', type=str, default='', help='path to training transformation csv folder')
parser.add_argument('--training-image-path', type=str, default='', help='path to folder containing training images')
parser.add_argument('--trained-models-dir', type=str, default='trained_models', help='path to trained models folder')
parser.add_argument('--trained-models-fn', type=str, default='checkpoint_adam', help='trained model filename')
# Optimization parameters 
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum constant')
parser.add_argument('--num-epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=16, help='training batch size')
parser.add_argument('--weight-decay', type=float, default=0, help='weight decay constant')
parser.add_argument('--seed', type=int, default=1, help='Pseudo-RNG seed')
# Model parameters
parser.add_argument('--geometric-model', type=str, default='affine', help='geometric model to be regressed at output: affine or tps')
parser.add_argument('--use-mse-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use MSE loss on tnf. parameters')
parser.add_argument('--feature-extraction-cnn', type=str, default='vgg', help='Feature extraction architecture: vgg/resnet101')
# Synthetic dataset parameters
parser.add_argument('--random-sample', type=str_to_bool, nargs='?', const=True, default=False, help='sample random transformations')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

# Seed
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# Download dataset if needed and set paths
if args.training_dataset == 'pascal':
    if args.training_image_path == '':
        download_pascal('datasets/pascal-voc11/')
        args.training_image_path = 'datasets/pascal-voc11/'
    if args.training_tnf_csv == '' and args.geometric_model=='affine':
        args.training_tnf_csv = 'training_data/pascal-synth-aff'
    elif args.training_tnf_csv == '' and args.geometric_model=='tps':
        args.training_tnf_csv = 'training_data/pascal-synth-tps'

# CNN model and loss
print('Creating CNN model...')

model = CNNGeometric(use_cuda=use_cuda,geometric_model=args.geometric_model,feature_extraction_cnn=args.feature_extraction_cnn)

if args.use_mse_loss:
    print('Using MSE loss...')
    loss = nn.MSELoss()
else:
    print('Using grid loss...')
    loss = TransformedGridLoss(use_cuda=use_cuda,geometric_model=args.geometric_model)


# Dataset and dataloader
dataset = SynthDataset(geometric_model=args.geometric_model,
                       csv_file=os.path.join(args.training_tnf_csv,'train.csv'),
                       training_image_path=args.training_image_path,
                       transform=NormalizeImageDict(['image']),
                       random_sample=args.random_sample)

dataloader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)

dataset_test = SynthDataset(geometric_model=args.geometric_model,
                            csv_file=os.path.join(args.training_tnf_csv,'test.csv'),
                            training_image_path=args.training_image_path,
                            transform=NormalizeImageDict(['image']),
                            random_sample=args.random_sample)

dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)


pair_generation_tnf = SynthPairTnf(geometric_model=args.geometric_model,use_cuda=use_cuda)

# Optimizer
optimizer = optim.Adam(model.FeatureRegression.parameters(), lr=args.lr)

# Train
if args.use_mse_loss:
    checkpoint_name = os.path.join(args.trained_models_dir,
                                   args.trained_models_fn + '_' + args.geometric_model + '_mse_loss' + args.feature_extraction_cnn + '.pth.tar')
else:
    checkpoint_name = os.path.join(args.trained_models_dir,
                                   args.trained_models_fn + '_' + args.geometric_model + '_grid_loss' + args.feature_extraction_cnn + '.pth.tar')
    
best_test_loss = float("inf")

print('Starting training...')

for epoch in range(1, args.num_epochs+1):
    train_loss = train(epoch,model,loss,optimizer,dataloader,pair_generation_tnf,log_interval=100)
    test_loss = test(model,loss,dataloader_test,pair_generation_tnf)
    
    # remember best loss
    is_best = test_loss < best_test_loss
    best_test_loss = min(test_loss, best_test_loss)
    save_checkpoint({
        'epoch': epoch + 1,
        'args': args,
        'state_dict': model.state_dict(),
        'best_test_loss': best_test_loss,
        'optimizer' : optimizer.state_dict(),
    }, is_best,checkpoint_name)

print('Done!')
