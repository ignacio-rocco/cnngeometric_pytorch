# CNNGeometric PyTorch implementation

![](http://www.di.ens.fr/willow/research/cnngeometric/images/code_teaser.png)

This is the implementation of the paper: 

I. Rocco, R. Arandjelović and J. Sivic. Convolutional neural network architecture for geometric matching. [[website](http://www.di.ens.fr/willow/research/cnngeometric/)][[CVPR version](https://arxiv.org/abs/1703.05593)][[Extended TPAMI version](https://hal.archives-ouvertes.fr/hal-01859616/file/cnngeometric_pami.pdf)]



## Dependencies 
See `requirements.txt`

## Demo 
Please see the `demo.py` script or the `demo_notebook.ipynb` Jupyter Notebook.

## Training
You can train the model using the `train.py` script in the following way:

```bash
python train.py  --geometric-model affine
```
For a full set of options, run `python train.py -h`.

##### Logging Configuration 

  - For now it is implemented to log on TensorBoard just scalars of train and val loss
  - It is possible to specify a --logdir as a parameter, otherwise the logging folder will be named as the checkpoint one with _tb_logs as suffix
  - N.B. If is intended to use as logdir a GCP bucket it is necessary to install Tensorflow 
  
## Evaluation
You can evaluate the trained models using the `eval.py` script in the following way:

```bash
python eval.py  --model-1 trained_models/best_streetview_checkpoint_adam_hom_grid_loss_PAMI.pth.tar --eval-dataset pf
```

You can also evaluate a two-stage model in the following way:

```bash
python eval.py --model-1 trained_models/best_streetview_checkpoint_adam_hom_grid_loss_PAMI.pth.tar --model-2 trained_models/best_streetview_checkpoint_adam_tps_grid_loss_PAMI.pth.tar --eval-dataset pf
```

The `eval.py` scripts implements the evaluation on the PF-Willow/PF-PASCAL/Caltech-101 and TSS datasets.  For a full set of options, run `python eval.py -h`.

### Trained models 

| Model | PF-Willow (PCK) |
| --- | --- | 
| [[Affine - VGG - StreetView]](http://www.di.ens.fr/willow/research/cnngeometric/trained_models/pytorch/best_streetview_checkpoint_adam_affine_grid_loss_PAMI.pth.tar) |  48.4 |
| [[Homography - VGG - StreetView]](http://www.di.ens.fr/willow/research/cnngeometric/trained_models/pytorch/best_streetview_checkpoint_adam_hom_grid_loss_PAMI.pth.tar) |  48.6 |
| [[TPS - VGG - StreetView]](http://www.di.ens.fr/willow/research/cnngeometric/trained_models/pytorch/best_streetview_checkpoint_adam_tps_grid_loss_PAMI.pth.tar) |  53.8 |


### BibTeX

If you use this code in your project, please cite us using:
```bibtex
@InProceedings{Rocco17,
  author = {Rocco, I. and Arandjelovi\'c, R. and Sivic, J.},
  title  = {Convolutional neural network architecture for geometric matching},
  booktitle = {{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition}},
  year = {2017},
}
```

or

```bibtex
@Article{Rocco18,
  author = {Rocco, I. and Arandjelovi\'c, R. and Sivic, J.},
  title  = {Convolutional neural network architecture for geometric matching},
  journal = {{IEEE Transactions on Pattern Analysis and Machine Intelligence}},
  number = {41},
  pages = {2553--2567},
  year = {2018},
}
```
