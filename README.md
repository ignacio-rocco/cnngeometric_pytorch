# CNNGeometric PyTorch implementation

![](http://www.di.ens.fr/willow/research/cnngeometric/images/teaser.png)

This is the implementation of the paper: 

I. Rocco, R. Arandjelović and J. Sivic. Convolutional neural network architecture for geometric matching. CVPR 2017 [[website](http://www.di.ens.fr/willow/research/cnngeometric/)][[arXiv](https://arxiv.org/abs/1703.05593)]

using PyTorch ([for MatConvNet implementation click here](https://github.com/ignacio-rocco/cnngeometric_matconvnet)).

If you use this code in your project, please cite use using:
````
@InProceedings{Rocco17,
  author       = "Rocco, I. and Arandjelovi\'c, R. and Sivic, J.",
  title        = "Convolutional neural network architecture for geometric matching",
  booktitle    = "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition",
  year         = "2017",
}
````

## Dependencies ###
  - Python 3
  - pytorch > 0.2.0, torchvision
  - numpy, skimage (included in conda)

## Getting started ###
  - demo.py demonstrates the results on the ProposalFlow dataset
  - train.py is the main training script
  - eval_pf.py evaluates on the ProposalFlow dataset
  
## Trained models ###

#### Using Streetview-synth dataset + VGG
  - [[Affine]](http://www.di.ens.fr/willow/research/cnngeometric/trained_models/pytorch/best_streetview_checkpoint_adam_affine_grid_loss.pth.tar), [[TPS]](http://www.di.ens.fr/willow/research/cnngeometric/trained_models/pytorch/best_streetview_checkpoint_adam_tps_grid_loss.pth.tar)
  - Results on PF: `PCK affine: 0.472`, `PCK tps: 0.513`, `PCK affine+tps: 0.572`

#### Using Pascal-synth dataset  + VGG
  - [[Affine]](http://www.di.ens.fr/willow/research/cnngeometric/trained_models/pytorch/best_pascal_checkpoint_adam_affine_grid_loss.pth.tar), [[TPS]](http://www.di.ens.fr/willow/research/cnngeometric/trained_models/pytorch/best_pascal_checkpoint_adam_tps_grid_loss.pth.tar)
  - Results on PF: `PCK affine: 0.478`, `PCK tps: 0.428`, `PCK affine+tps: 0.568`

#### Using Pascal-synth dataset  + ResNet-101
  - [[Affine]](http://www.di.ens.fr/willow/research/cnngeometric/trained_models/pytorch/best_pascal_checkpoint_adam_affine_grid_loss_resnet_random.pth.tar), [[TPS]](http://www.di.ens.fr/willow/research/cnngeometric/trained_models/pytorch/best_pascal_checkpoint_adam_tps_grid_loss_resnet_random.pth.tar)
  - Results on PF: `PCK affine: 0.559`, `PCK tps: 0.582`, `PCK affine+tps: 0.676`
