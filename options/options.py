import argparse
from util.torch_util import str_to_bool

class ArgumentParser():
    def __init__(self,mode='train'):
        self.parser = argparse.ArgumentParser(description='CNNGeometric PyTorch implementation')
        
        self.add_cnn_model_parameters()
        if mode=='train':
            self.add_train_parameters()
            self.add_synth_dataset_parameters()
            self.add_base_train_parameters()
        elif mode=='eval':
            self.add_eval_parameters()
            self.add_base_eval_parameters()

    def add_cnn_model_parameters(self):
        model_params = self.parser.add_argument_group('model')
        # Model parameters
        model_params.add_argument('--feature-extraction-cnn', type=str, default='vgg', help='feature extraction CNN model architecture: vgg/resnet101')
        model_params.add_argument('--feature-extraction-last-layer', type=str, default='', help='feature extraction CNN last layer')
        model_params.add_argument('--fr-kernel-sizes', nargs='+', type=int, default=[7,5,5], help='kernels sizes in feat.reg. conv layers')
        model_params.add_argument('--fr-channels', nargs='+', type=int, default=[225,128,64], help='channels in feat. reg. conv layers')
        model_params.add_argument('--matching-type', type=str, default='correlation', help='correlation/subtraction/concatenation')
        model_params.add_argument('--normalize-matches', type=str_to_bool, nargs='?', const=True, default=True, help='perform L2 normalization')        

    def add_base_train_parameters(self):
        base_params = self.parser.add_argument_group('base')
        # Image size
        base_params.add_argument('--image-size', type=int, default=240, help='image input size')
        # Pre-trained model file
        base_params.add_argument('--model', type=str, default='', help='Pre-trained model filename')
        
        base_params.add_argument('--num-workers', type=int, default=4, help='number of workers')

    def add_base_eval_parameters(self):
        base_params = self.parser.add_argument_group('base')
        # Image size
        base_params.add_argument('--image-size', type=int, default=240, help='image input size')
        # Pre-trained model file
        base_params.add_argument('--model-1', type=str, default='', help='Trained model - stage 1')
        base_params.add_argument('--model-2', type=str, default='', help='Trained model - stage 2')
        # Number of stages
        base_params.add_argument('--num-of-iters', type=int, default=1, help='number of stages to use recursively')
    
    def add_synth_dataset_parameters(self):
        dataset_params = self.parser.add_argument_group('dataset')
        # Dataset parameters
        dataset_params.add_argument('--dataset-csv-path', type=str, default='', help='path to training transformation csv folder')
        dataset_params.add_argument('--dataset-image-path', type=str, default='', help='path to folder containing training images')
        # Random synth dataset parameters
        dataset_params.add_argument('--four-point-hom', type=str_to_bool, nargs='?', const=True, default=True, help='use 4 pt parametrization for homography')
        dataset_params.add_argument('--random-sample', type=str_to_bool, nargs='?', const=True, default=True, help='sample random transformations')
        dataset_params.add_argument('--random-t', type=float, default=0.5, help='random transformation translation')
        dataset_params.add_argument('--random-s', type=float, default=0.5, help='random transformation translation')
        dataset_params.add_argument('--random-alpha', type=float, default=1/6, help='random transformation translation')
        dataset_params.add_argument('--random-t-tps', type=float, default=0.4, help='random transformation translation')        
        

    def add_train_parameters(self):
        train_params = self.parser.add_argument_group('train')
        # Optimization parameters 
        train_params.add_argument('--lr', type=float, default=0.001, help='learning rate')
        train_params.add_argument('--lr_scheduler', type=str_to_bool,
                        nargs='?', const=True, default=True,
                        help='Bool (default True), whether to use a decaying lr_scheduler')
        train_params.add_argument('--lr_max_iter', type=int, default=1000,
                        help='Number of steps between lr starting value and 1e-6 '
                             '(lr default min) when choosing lr_scheduler')
        train_params.add_argument('--momentum', type=float, default=0.9, help='momentum constant')
        train_params.add_argument('--num-epochs', type=int, default=20, help='number of training epochs')
        train_params.add_argument('--batch-size', type=int, default=16, help='training batch size')
        train_params.add_argument('--weight-decay', type=float, default=0, help='weight decay constant')
        train_params.add_argument('--seed', type=int, default=1, help='Pseudo-RNG seed')
        train_params.add_argument('--use-mse-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use MSE loss on tnf. parameters')        
        train_params.add_argument('--geometric-model', type=str, default='affine', help='affine/hom/tps')
        # Trained model parameters
        train_params.add_argument('--trained-model-fn', type=str, default='checkpoint_adam', help='trained model filename')
        train_params.add_argument('--trained-model-dir', type=str, default='trained_models', help='path to trained models folder')
        # Dataset name (used for loading defaults)
        train_params.add_argument('--training-dataset', type=str, default='pascal', help='dataset to use for training')
        # Limit train/test dataset sizes
        train_params.add_argument('--train-dataset-size', type=int, default=0, help='train dataset size limit')
        train_params.add_argument('--test-dataset-size', type=int, default=0, help='test dataset size limit')
        # Parts of model to train
        train_params.add_argument('--train-fe', type=str_to_bool, nargs='?', const=True, default=True, help='Train feature extraction')
        train_params.add_argument('--train-fr', type=str_to_bool, nargs='?', const=True, default=True, help='Train feature regressor')
        train_params.add_argument('--train-bn', type=str_to_bool, nargs='?', const=True, default=True, help='train batch-norm layers')
        train_params.add_argument('--fe-finetune-params',  nargs='+', type=str, default=[''], help='String indicating the F.Ext params to finetune')
        train_params.add_argument('--update-bn-buffers', type=str_to_bool, nargs='?', const=True, default=False, help='Update batch norm running mean and std')        
        # Train with occlusion
        train_params.add_argument('--occlusion-factor', type=float, default=0, help='occlusion factor for training')
        # log parameters
        train_params.add_argument('--log_interval', type=int, default=100,
                        help='Number of iterations between logs')
        train_params.add_argument('--log_dir', type=str, default='',
                        help='If unspecified log_dir will be set to'
                             '<trained_models_dir>/<trained_models_fn>/')

    def add_eval_parameters(self):
        eval_params = self.parser.add_argument_group('eval')
        # Evaluation parameters
        eval_params.add_argument('--eval-dataset', type=str, default='pf', help='pf/caltech/tss')
        eval_params.add_argument('--eval-dataset-path', type=str, default='', help='Path to PF dataset')
        eval_params.add_argument('--flow-output-dir', type=str, default='results/', help='flow output dir')
        eval_params.add_argument('--pck-alpha', type=float, default=0.1, help='pck margin factor alpha')
        eval_params.add_argument('--eval-metric', type=str, default='pck', help='pck/distance')
        eval_params.add_argument('--tps-reg-factor', type=float, default=0.0, help='regularisation factor for tps tnf')
        eval_params.add_argument('--batch-size', type=int, default=16, help='batch size (only GPU)')
        

        
    def parse(self,arg_str=None):
        if arg_str is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(arg_str.split())
        arg_groups = {}
        for group in self.parser._action_groups:
            group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
            arg_groups[group.title]=group_dict
        return (args,arg_groups)

        