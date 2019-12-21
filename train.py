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
from data.download_datasets import download_pascal

from geotnf.transformation import SynthPairTnf

from image.normalization import NormalizeImageDict

from util.train_test_fn import train, validate_model
from util.torch_util import save_checkpoint, str_to_bool

from options.options import ArgumentParser


"""

Script to evaluate a trained model as presented in the CNNGeometric TPAMI paper
on the PF/PF-pascal/Caltech-101 and TSS datasets

"""

def main():

    args,arg_groups = ArgumentParser(mode='train').parse()
    print(args)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    # Seed
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # Download dataset if needed and set paths
    if args.training_dataset == 'pascal':

        if args.dataset_image_path == '' and not os.path.exists('datasets/pascal-voc11/TrainVal'):
            download_pascal('datasets/pascal-voc11/')

        if args.dataset_image_path == '':
            args.dataset_image_path = 'datasets/pascal-voc11/'

        args.dataset_csv_path = 'training_data/pascal-random'        


    # CNN model and loss
    print('Creating CNN model...')
    if args.geometric_model=='affine':
        cnn_output_dim = 6
    elif args.geometric_model=='hom' and args.four_point_hom:
        cnn_output_dim = 8
    elif args.geometric_model=='hom' and not args.four_point_hom:
        cnn_output_dim = 9
    elif args.geometric_model=='tps':
        cnn_output_dim = 18

    model = CNNGeometric(use_cuda=use_cuda,
                         output_dim=cnn_output_dim,
                         **arg_groups['model'])

    if args.geometric_model=='hom' and not args.four_point_hom:
        init_theta = torch.tensor([1,0,0,0,1,0,0,0,1], device = device)
        model.FeatureRegression.linear.bias.data+=init_theta

    if args.geometric_model=='hom' and args.four_point_hom:
        init_theta = torch.tensor([-1, -1, 1, 1, -1, 1, -1, 1], device = device)
        model.FeatureRegression.linear.bias.data+=init_theta

    if args.use_mse_loss:
        print('Using MSE loss...')
        loss = nn.MSELoss()
    else:
        print('Using grid loss...')
        loss = TransformedGridLoss(use_cuda=use_cuda,
                                   geometric_model=args.geometric_model)

    # Initialize Dataset objects
    dataset = SynthDataset(geometric_model=args.geometric_model,
               dataset_csv_path=args.dataset_csv_path,
               dataset_csv_file='train.csv',
			   dataset_image_path=args.dataset_image_path,
			   transform=NormalizeImageDict(['image']),
			   random_sample=args.random_sample)

    dataset_val = SynthDataset(geometric_model=args.geometric_model,
                   dataset_csv_path=args.dataset_csv_path,
                   dataset_csv_file='val.csv',
			       dataset_image_path=args.dataset_image_path,
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
        ckpt = args.trained_model_fn + '_' + args.geometric_model + '_mse_loss' + args.feature_extraction_cnn
        checkpoint_path = os.path.join(args.trained_model_dir,
                                       args.trained_model_fn,
                                       ckpt + '.pth.tar')
    else:
        ckpt = args.trained_model_fn + '_' + args.geometric_model + '_grid_loss' + args.feature_extraction_cnn
        checkpoint_path = os.path.join(args.trained_model_dir,
                                       args.trained_model_fn,
                                       ckpt + '.pth.tar')
    if not os.path.exists(args.trained_model_dir):
        os.mkdir(args.trained_model_dir)

    # Set up TensorBoard writer
    if not args.log_dir:
        tb_dir = os.path.join(args.trained_model_dir, args.trained_model_fn + '_tb_logs')
    else:
        tb_dir = os.path.join(args.log_dir, args.trained_model_fn + '_tb_logs')

    logs_writer = SummaryWriter(tb_dir)
    # add graph, to do so we have to generate a dummy input to pass along with the graph
    dummy_input = {'source_image': torch.rand([args.batch_size, 3, 240, 240], device = device),
                   'target_image': torch.rand([args.batch_size, 3, 240, 240], device = device),
                   'theta_GT': torch.rand([16, 2, 3], device = device)}

    logs_writer.add_graph(model, dummy_input)

    # Start of training
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
