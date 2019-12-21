from __future__ import print_function, division
import os
from os.path import exists
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data.pf_dataset import PFDataset, PFPascalDataset
from data.caltech_dataset import CaltechDataset
from data.tss_dataset import TSSDataset
from data.download_datasets import *
from geotnf.point_tnf import *
from geotnf.transformation import GeometricTnf
from image.normalization import NormalizeImageDict
from model.cnn_geometric_model import CNNGeometric
from options.options import ArgumentParser
from util.torch_util import BatchTensorToVars, str_to_bool
from util.eval_util import pck_metric, area_metrics, flow_metrics, compute_metric
from util.dataloader import default_collate

"""

Script to evaluate a trained model as presented in the CNNGeometric TPAMI paper
on the PF/PF-pascal/Caltech-101 and TSS datasets

"""

def main():

    # Argument parsing
    args,arg_groups = ArgumentParser(mode='eval').parse()
    print(args)

    # check provided models and deduce if single/two-stage model should be used
    two_stage = args.model_2 != ''
     

    if args.eval_dataset_path == '' and args.eval_dataset == 'pf':
        args.eval_dataset_path = 'datasets/proposal-flow-willow/'

    if args.eval_dataset_path == '' and args.eval_dataset == 'pf-pascal':
        args.eval_dataset_path = 'datasets/proposal-flow-pascal/'

    if args.eval_dataset_path == '' and args.eval_dataset == 'caltech':
        args.eval_dataset_path = 'datasets/caltech-101/'
        
    if args.eval_dataset_path == '' and args.eval_dataset == 'tss':
        args.eval_dataset_path = 'datasets/tss/'

    use_cuda = torch.cuda.is_available()

    # Download dataset if needed
    if args.eval_dataset == 'pf' and not exists(args.eval_dataset_path):
        download_PF_willow(args.eval_dataset_path)

    elif args.eval_dataset == 'pf-pascal' and not exists(args.eval_dataset_path):
        download_PF_pascal(args.eval_dataset_path)

    elif args.eval_dataset == 'caltech' and not exists(args.eval_dataset_path):
        download_caltech(args.eval_dataset_path)

    elif args.eval_dataset == 'tss' and not exists(args.eval_dataset_path):
        download_TSS(args.eval_dataset_path)


    print('Creating CNN model...')


    def create_model(model_filename):
        checkpoint = torch.load(model_filename, map_location=lambda storage, loc: storage)
        checkpoint['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
        output_size = checkpoint['state_dict']['FeatureRegression.linear.bias'].size()[0]

        if output_size == 6:
            geometric_model = 'affine'

        elif output_size == 8 or output_size == 9:
            geometric_model = 'hom'
        else: 
            geometric_model = 'tps'

        model = CNNGeometric(use_cuda=use_cuda,
                             output_dim=output_size,
                             **arg_groups['model'])

        for name, param in model.FeatureExtraction.state_dict().items():
            if not name.endswith('num_batches_tracked'):
                model.FeatureExtraction.state_dict()[name].copy_(checkpoint['state_dict']['FeatureExtraction.' + name])    

        for name, param in model.FeatureRegression.state_dict().items():
            if not name.endswith('num_batches_tracked'):
                model.FeatureRegression.state_dict()[name].copy_(checkpoint['state_dict']['FeatureRegression.' + name])

        return (model,geometric_model)

    # Load model for stage 1
    model_1, geometric_model_1 = create_model(args.model_1)

    if two_stage:
        # Load model for stage 2
        model_2, geometric_model_2 = create_model(args.model_2)
    else:
        model_2,geometric_model_2 = None, None

    #import pdb; pdb.set_trace()

    print('Creating dataset and dataloader...')

    # Dataset and dataloader
    if args.eval_dataset == 'pf':  
        Dataset = PFDataset
        collate_fn = default_collate
        csv_file = 'test_pairs_pf.csv'

    if args.eval_dataset == 'pf-pascal':  
        Dataset = PFPascalDataset
        collate_fn = default_collate
        csv_file = 'all_pairs_pf_pascal.csv'    

    elif args.eval_dataset == 'caltech':
        Dataset = CaltechDataset
        collate_fn = default_collate
        csv_file = 'test_pairs_caltech_with_category.csv'

    elif args.eval_dataset == 'tss':
        Dataset = TSSDataset
        collate_fn = default_collate
        csv_file = 'test_pairs_tss.csv'
        
    cnn_image_size=(args.image_size,args.image_size)

    dataset = Dataset(csv_file = os.path.join(args.eval_dataset_path, csv_file),
                      dataset_path = args.eval_dataset_path,
                      transform = NormalizeImageDict(['source_image','target_image']),
                      output_size = cnn_image_size)

    if use_cuda:
        batch_size = args.batch_size

    else:
        batch_size = 1

    dataloader = DataLoader(dataset, 
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers=0,
                            collate_fn = collate_fn)

    batch_tnf = BatchTensorToVars(use_cuda = use_cuda)

    if args.eval_dataset == 'pf' or args.eval_dataset == 'pf-pascal':  
        metric = 'pck'

    elif args.eval_dataset == 'caltech':
        metric = 'area'

    elif args.eval_dataset == 'tss':
        metric = 'flow'
        
    model_1.eval()

    if two_stage:
        model_2.eval()

    print('Starting evaluation...')
        
    stats=compute_metric(metric,
                         model_1,
                         geometric_model_1,
                         model_2,
                         geometric_model_2,
                         dataset,
                         dataloader,
                         batch_tnf,
                         batch_size,
                         args)

if __name__ == '__main__':
    main()
