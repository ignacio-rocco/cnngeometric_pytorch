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
See `requirements.txt`

## Getting started ###
  - demo.py demonstrates the results on the ProposalFlow dataset
  - train.py is the main training script
  - eval_pf.py evaluates on the ProposalFlow dataset
  
## Logging Configuration ###

  - For now it is implemented to log on TensorBoard just scalars of train and val loss
  - It is possible to specify a --logdir as a parameter, otherwise the logging folder will be named as the checkpoint one with _tb_logs as suffix
  - N.B. If is intended to use as logdir a GCP bucket it is necessary to install Tensorflow 
  
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

#### Using a custom dataset
  - It is possible to use a custom dataset, in order to do so is necessary to create a custom Dataset object and modify the serving function (ex. SynthPairTnf/CoupledPairTnf)
  - In the case of the CoupledPairTnf class, the dataset was in the format ['image_a', 'image_b', 'vertices_a', *theta_components] where theta is the affine matrix
  - N.B. when using a custom dataset make sure that bounding boxes and points contained are normalized over the dimensions of the image, transformations as well should be computed from normalized points
  - Example of coupled dataset line:
  
  
    image_a, image_b, vertices_a, A22, A21, A12, A11, ty, tx  
    image_a.jpg,image_b.png,"[(0.499, 0.094), (0.810, 0.100), (0.795, 0.437), (0.485, 0.430)]",1.0017,-0.0179,0.0390,0.9875,-0.0074,-0.0047
