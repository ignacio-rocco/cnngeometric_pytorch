from __future__ import print_function, division
import argparse
import os
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.cnn_geometric_model import CNNGeometric
from model.loss import TransformedGridLoss

from data.synth_dataset import SynthDataset
from data.coupled_dataset import CoupledDataset
from data.download_datasets import download_pascal

from geotnf.transformation import SynthPairTnf
from geotnf.transformation import CoupledPairTnf

from image.normalization import NormalizeImageDict

from util.train_test_fn import train, validate_model
from util.torch_util import save_checkpoint, str_to_bool

"""

Script to train the model as presented in the CNNGeometric CVPR'17 paper
using synthetically warped image pairs and strong supervision

"""


def parse_flags():
    """
    Fn to parse arguments to pass to main func

    :return args: Object which attributes are the flags values accessible through args.flag
    """

    # Argument parsing
    parser = argparse.ArgumentParser(description='CNNGeometric PyTorch implementation')
    # Paths
    parser.add_argument('--training_dataset', type=str, default='pascal',
                        help='dataset to use for training')
    parser.add_argument('--training_tnf_csv', type=str, default='',
                        help='path to training transformation csv folder')
    parser.add_argument('--training_image_path', type=str, default='',
                        help='path to folder containing training images')
    parser.add_argument('--trained_models_dir', type=str, default='trained_models',
                        help='path to trained models folder')
    parser.add_argument('--trained_models_fn', type=str, default='checkpoint_adam',
                        help='trained model filename')
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_scheduler', type=str_to_bool,
                        nargs='?', const=True, default=True,
                        help='Bool (default True), whether to use a decaying lr_scheduler')
    parser.add_argument('--lr_max_iter', type=int, default=1000,
                        help='Number of steps between lr starting value and 1e-6 '
                             '(lr default min) when choosing lr_scheduler')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum constant')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay constant')
    parser.add_argument('--seed', type=int, default=1,
                        help='Pseudo-RNG seed')
    # Model parameters
    parser.add_argument('--geometric_model', type=str, default='affine',
                        help='geometric model to be regressed at output: affine or tps')
    parser.add_argument('--use_mse_loss', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Use MSE loss on tnf. parameters')
    parser.add_argument('--feature_extraction_cnn', type=str, default='vgg',
                        help='Feature extraction architecture: vgg/resnet101')
    # Synthetic dataset parameters
    parser.add_argument('--random_sample', type=str_to_bool, nargs='?', const=True, default=False,
                        help='sample random transformations')
    parser.add_argument('--coupled_dataset', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Whether csv dataset contains already pair of images')
    # log parameters
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Number of iterations between logs')
    parser.add_argument('--log_dir', type=str, default='',
                        help='If unspecified log_dir will be set to'
                             '<trained_models_dir>/<trained_models_fn>/')

    return parser.parse_args()


def main():

    args = parse_flags()

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

        if args.training_tnf_csv == '' and args.geometric_model == 'affine':

            args.training_tnf_csv = 'training_data/pascal-synth-aff'

        elif args.training_tnf_csv == '' and args.geometric_model == 'tps':

            args.training_tnf_csv = 'training_data/pascal-synth-tps'

    # CNN model and loss
    print('Creating CNN model...')

    model = CNNGeometric(use_cuda=use_cuda,
                         geometric_model=args.geometric_model,
                         feature_extraction_cnn=args.feature_extraction_cnn)

    if args.use_mse_loss:
        print('Using MSE loss...')
        loss = nn.MSELoss()
    else:
        print('Using grid loss...')
        loss = TransformedGridLoss(use_cuda=use_cuda,
                                   geometric_model=args.geometric_model)

    # Initialize csv paths
    train_csv_path_list = glob(os.path.join(args.training_tnf_csv, '*train.csv'))
    if len(train_csv_path_list) > 1:
        print("!!!!WARNING!!!! multiple train csv files found, using first in glob order")
    elif not len(train_csv_path_list):
        raise FileNotFoundError("No training csv where found in the specified path!!!")

    train_csv_path = train_csv_path_list[0]

    val_csv_path_list = glob(os.path.join(args.training_tnf_csv, '*val.csv'))
    if len(val_csv_path_list) > 1:
        print("!!!!WARNING!!!! multiple train csv files found, using first in glob order")
    elif not len(val_csv_path_list):
        raise FileNotFoundError("No training csv where found in the specified path!!!")

    val_csv_path = val_csv_path_list[0]

    # Initialize Dataset objects
    if args.coupled_dataset:
        # Dataset  for train and val if dataset is already coupled
        dataset = CoupledDataset(geometric_model=args.geometric_model,
                                 csv_file=train_csv_path,
                                 training_image_path=args.training_image_path,
                                 transform=NormalizeImageDict(['image_a', 'image_b']),
                                 random_sample=args.random_sample)

        dataset_val = CoupledDataset(geometric_model=args.geometric_model,
                                     csv_file=val_csv_path,
                                     training_image_path=args.training_image_path,
                                     transform=NormalizeImageDict(['image_a', 'image_b']),
                                     random_sample=args.random_sample)

        # Set Tnf pair generation func
        pair_generation_tnf = CoupledPairTnf(use_cuda=use_cuda)

    else:
        # Standard Dataset for train and val
        dataset = SynthDataset(geometric_model=args.geometric_model,
                               csv_file=train_csv_path,
                               training_image_path=args.training_image_path,
                               transform=NormalizeImageDict(['image']),
                               random_sample=args.random_sample)

        dataset_val = SynthDataset(geometric_model=args.geometric_model,
                                   csv_file=val_csv_path,
                                   training_image_path=args.training_image_path,
                                   transform=NormalizeImageDict(['image']),
                                   random_sample=args.random_sample)

        # Set Tnf pair generation func
        pair_generation_tnf = SynthPairTnf(geometric_model=args.geometric_model,
                                           use_cuda=use_cuda)

    # Initialize DataLoaders
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4)

    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size,
                                shuffle=True, num_workers=4)

    # Optimizer and eventual scheduler
    optimizer = optim.Adam(model.FeatureRegression.parameters(), lr=args.lr)

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.lr_max_iter,
                                                               eta_min=1e-6)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    else:
        scheduler = False

    # Train

    # Set up names for checkpoints
    if args.use_mse_loss:
        ckpt = args.trained_models_fn + '_' + args.geometric_model + '_mse_loss' + args.feature_extraction_cnn
        checkpoint_path = os.path.join(args.trained_models_dir,
                                       args.trained_models_fn,
                                       ckpt + '.pth.tar')
    else:
        ckpt = args.trained_models_fn + '_' + args.geometric_model + '_grid_loss' + args.feature_extraction_cnn
        checkpoint_path = os.path.join(args.trained_models_dir,
                                       args.trained_models_fn,
                                       ckpt + '.pth.tar')
    if not os.path.exists(args.trained_models_dir):
        os.mkdir(args.trained_models_dir)

    # Set up TensorBoard writer
    if not args.log_dir:
        tb_dir = os.path.join(args.trained_models_dir, args.trained_models_fn + '_tb_logs')
    else:
        tb_dir = os.path.join(args.log_dir, args.trained_models_fn + '_tb_logs')

    logs_writer = SummaryWriter(tb_dir)
    # add graph, to do so we have to generate a dummy input to pass along with the graph
    dummy_input = {'source_image': torch.rand([args.batch_size, 3, 240, 240]),
                   'target_image': torch.rand([args.batch_size, 3, 240, 240]),
                   'theta_GT': torch.rand([16, 2, 3])}

    logs_writer.add_graph(model, dummy_input)

    #                START OF TRAINING                 #
    print('Starting training...')

    best_val_loss = float("inf")

    for epoch in range(1, args.num_epochs+1):

        # we don't need the average epoch loss so we assign it to _
        _ = train(epoch, model, loss, optimizer,
                  dataloader, pair_generation_tnf,
                  log_interval=args.log_interval,
                  scheduler=scheduler,
                  tb_writer=logs_writer)

        val_loss = validate_model(model, loss,
                                  dataloader_val, pair_generation_tnf,
                                  epoch, logs_writer)

        # remember best loss
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        save_checkpoint({
                         'epoch': epoch + 1,
                         'args': args,
                         'state_dict': model.state_dict(),
                         'best_val_loss': best_val_loss,
                         'optimizer': optimizer.state_dict(),
                         },
                        is_best, checkpoint_path)

    logs_writer.close()
    print('Done!')


if __name__ == '__main__':
    main()
