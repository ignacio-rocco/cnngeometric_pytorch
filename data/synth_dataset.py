from __future__ import print_function, division
import torch
import os
from os.path import exists, join, basename
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf
from torch.autograd import Variable

class SynthDataset(Dataset):
    """
    
    Synthetically transformed pairs dataset for training with strong supervision
    
    Args:
            csv_file (string): Path to the csv file with image names and transformations.
            training_image_path (string): Directory with all the images.
            transform (callable): Transformation for post-processing the training pair (eg. image normalization)
            
    Returns:
            Dict: {'image': full dataset image, 'theta': desired transformation}
            
    """

    def __init__(self, csv_file, training_image_path, output_size=(480,640), geometric_model='affine', transform=None,
                 random_sample=False, random_t=0.5, random_s=0.5, random_alpha=1/6, random_t_tps=0.4):
        self.random_sample = random_sample
        self.random_t = random_t
        self.random_t_tps = random_t_tps
        self.random_alpha = random_alpha
        self.random_s = random_s
        self.out_h, self.out_w = output_size
        # read csv file
        self.train_data = pd.read_csv(csv_file)
        self.img_names = self.train_data.iloc[:,0]
        self.theta_array = self.train_data.iloc[:, 1:].as_matrix().astype('float')
        # copy arguments
        self.training_image_path = training_image_path
        self.transform = transform
        self.geometric_model = geometric_model
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda = False) 
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # read image
        img_name = os.path.join(self.training_image_path, self.img_names[idx])
        image = io.imread(img_name)
        
        # read theta
        if self.random_sample==False:
            theta = self.theta_array[idx, :]

            if self.geometric_model=='affine':
                # reshape theta to 2x3 matrix [A|t] where 
                # first row corresponds to X and second to Y
                theta = theta[[3,2,5,1,0,4]].reshape(2,3)
            elif self.geometric_model=='tps':
                theta = np.expand_dims(np.expand_dims(theta,1),2)
        else:
            if self.geometric_model=='affine':
                alpha = (np.random.rand(1)-0.5)*2*np.pi*self.random_alpha
                theta = np.random.rand(6)
                theta[[2,5]]=(theta[[2,5]]-0.5)*2*self.random_t
                theta[0]=(1+(theta[0]-0.5)*2*self.random_s)*np.cos(alpha)
                theta[1]=(1+(theta[1]-0.5)*2*self.random_s)*(-np.sin(alpha))
                theta[3]=(1+(theta[3]-0.5)*2*self.random_s)*np.sin(alpha)
                theta[4]=(1+(theta[4]-0.5)*2*self.random_s)*np.cos(alpha)
                theta = theta.reshape(2,3)
            if self.geometric_model=='tps':
                theta = np.array([-1 , -1 , -1 , 0 , 0 , 0 , 1 , 1 , 1 , -1 , 0 , 1 , -1 , 0 , 1 , -1 , 0 , 1])
                theta = theta+(np.random.rand(18)-0.5)*2*self.random_t_tps               
        
        # make arrays float tensor for subsequent processing
        image = torch.Tensor(image.astype(np.float32))
        theta = torch.Tensor(theta.astype(np.float32))
        
        # permute order of image to CHW
        image = image.transpose(1,2).transpose(0,1)
                
        # Resize image using bilinear sampling with identity affine tnf
        if image.size()[0]!=self.out_h or image.size()[1]!=self.out_w:
            image = self.affineTnf(Variable(image.unsqueeze(0),requires_grad=False)).data.squeeze(0)
                
        sample = {'image': image, 'theta': theta}
        
        if self.transform:
            sample = self.transform(sample)

        return sample