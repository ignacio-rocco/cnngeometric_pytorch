from __future__ import print_function, division
import os
import torch
from torch.autograd import Variable
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf
from geotnf.flow import read_flo_file

class TSSDataset(Dataset):
    
    """
    
    TSS image pair dataset
    
    http://taniai.space/projects/cvpr16_dccs/
    

    Args:
        csv_file (string): Path to the csv file with image names and annotation files.
        dataset_path (string): Directory with the images.
        output_size (2-tuple): Desired output size
        transform (callable): Transformation for post-processing the training pair (eg. image normalization)
        
    """

    def __init__(self, csv_file, dataset_path,output_size=(240,240),transform=None):

        self.out_h, self.out_w = output_size
        self.pairs = pd.read_csv(csv_file)
        self.img_A_names = self.pairs.iloc[:,0]
        self.img_B_names = self.pairs.iloc[:,1]
        self.flow_direction = self.pairs.iloc[:, 2].values.astype('int')
        self.flip_img_A = self.pairs.iloc[:, 3].values.astype('int')
        self.pair_category = self.pairs.iloc[:, 4].values.astype('int')
        self.dataset_path = dataset_path         
        self.transform = transform
        # no cuda as dataset is called from CPU threads in dataloader and produces confilct
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda = False) 
              
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # get pre-processed images
        flip_img_A = self.flip_img_A[idx]
        image_A,im_size_A = self.get_image(self.img_A_names,idx,flip_img_A)
        image_B,im_size_B = self.get_image(self.img_B_names,idx)

        # get flow output path
        flow_path = self.get_GT_flow_relative_path(idx)

        sample = {'source_image': image_A, 'target_image': image_B, 'source_im_size': im_size_A, 'target_im_size': im_size_B, 'flow_path': flow_path}
        
        # # get ground-truth flow
        # flow = self.get_GT_flow(idx)
        
        # sample = {'source_image': image_A, 'target_image': image_B, 'source_im_size': im_size_A, 'target_im_size': im_size_B, 'flow_GT': flow}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self,img_name_list,idx,flip=False):
        img_name = os.path.join(self.dataset_path, img_name_list[idx])
        image = io.imread(img_name)
        
        # if grayscale convert to 3-channel image 
        if image.ndim==2:
            image=np.repeat(np.expand_dims(image,2),axis=2,repeats=3)
            
        # flip horizontally if needed
        if flip:
            image=np.flip(image,1)
            
        # get image size
        im_size = np.asarray(image.shape)
        
        # convert to torch Variable
        image = np.expand_dims(image.transpose((2,0,1)),0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image,requires_grad=False)
        
        # Resize image using bilinear sampling with identity affine tnf
        image = self.affineTnf(image_var).data.squeeze(0)
        
        im_size = torch.Tensor(im_size.astype(np.float32))
        
        return (image, im_size)
    
    def get_GT_flow(self,idx):
        img_folder = os.path.dirname(self.img_A_names[idx])
        flow_dir = self.flow_direction[idx]
        flow_file = 'flow'+str(flow_dir)+'.flo'
        flow_file_path = os.path.join(self.dataset_path, img_folder , flow_file)
        
        flow = torch.FloatTensor(read_flo_file(flow_file_path))

        return flow
    
    def get_GT_flow_relative_path(self,idx):
        img_folder = os.path.dirname(self.img_A_names[idx])
        flow_dir = self.flow_direction[idx]
        flow_file = 'flow'+str(flow_dir)+'.flo'
        flow_file_path = os.path.join(img_folder , flow_file)
        
        return flow_file_path
        