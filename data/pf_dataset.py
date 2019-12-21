from __future__ import print_function, division
import os
import torch
from torch.autograd import Variable
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf

class PFDataset(Dataset):
    
    """
    
    Proposal Flow image pair dataset
    

    Args:
        csv_file (string): Path to the csv file with image names and transformations.
        dataset_path (string): Directory with the images.
        output_size (2-tuple): Desired output size
        transform (callable): Transformation for post-processing the training pair (eg. image normalization)
        
    """

    def __init__(self, csv_file, dataset_path, output_size=(240,240), transform=None):

        self.out_h, self.out_w = output_size
        self.pairs = pd.read_csv(csv_file)
        self.img_A_names = self.pairs.iloc[:,0]
        self.img_B_names = self.pairs.iloc[:,1]
        self.point_A_coords = self.pairs.iloc[:, 2:22].values.astype('float')
        self.point_B_coords = self.pairs.iloc[:, 22:].values.astype('float')
        self.dataset_path = dataset_path         
        self.transform = transform
        # no cuda as dataset is called from CPU threads in dataloader and produces confilct
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda = False) 
              
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # get pre-processed images
        image_A,im_size_A = self.get_image(self.img_A_names,idx)
        image_B,im_size_B = self.get_image(self.img_B_names,idx)

        # get pre-processed point coords
        point_A_coords = self.get_points(self.point_A_coords,idx)
        point_B_coords = self.get_points(self.point_B_coords,idx)
        
        # compute PCK reference length L_pck (equal to max bounding box side in image_A)
        L_pck = torch.FloatTensor([torch.max(point_A_coords.max(1)[0]-point_A_coords.min(1)[0])])
                
        sample = {'source_image': image_A, 'target_image': image_B, 'source_im_size': im_size_A, 'target_im_size': im_size_B, 'source_points': point_A_coords, 'target_points': point_B_coords, 'L_pck': L_pck}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self,img_name_list,idx):
        img_name = os.path.join(self.dataset_path, img_name_list[idx])
        image = io.imread(img_name)
        
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
    
    def get_points(self,point_coords_list,idx):
        point_coords = point_coords_list[idx, :].reshape(2,10)

#        # swap X,Y coords, as the the row,col order (Y,X) is used for computations
#        point_coords = point_coords[[1,0],:]

        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords
    
    
class PFPascalDataset(Dataset):
    
    """
    
    Proposal Flow image pair dataset
    

    Args:
        csv_file (string): Path to the csv file with image names and transformations.
        dataset_path (string): Directory with the images.
        output_size (2-tuple): Desired output size
        transform (callable): Transformation for post-processing the training pair (eg. image normalization)
        
    """

    def __init__(self, csv_file, dataset_path, output_size=(240,240), transform=None, category=None):

        self.category_names=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
        self.out_h, self.out_w = output_size
        self.pairs = pd.read_csv(csv_file)
        self.category = self.pairs.iloc[:,2].values.astype('float')
        if category is not None:
            cat_idx = np.nonzero(self.category==category)[0]
            self.category=self.category[cat_idx]
            self.pairs=self.pairs.iloc[cat_idx,:]
        self.img_A_names = self.pairs.iloc[:,0]
        self.img_B_names = self.pairs.iloc[:,1]
        self.point_A_coords = self.pairs.iloc[:, 3:5]
        self.point_B_coords = self.pairs.iloc[:, 5:]
        self.dataset_path = dataset_path         
        self.transform = transform
        # no cuda as dataset is called from CPU threads in dataloader and produces confilct
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda = False) 
              
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # get pre-processed images
        image_A,im_size_A = self.get_image(self.img_A_names,idx)
        image_B,im_size_B = self.get_image(self.img_B_names,idx)

        # get pre-processed point coords
        point_A_coords = self.get_points(self.point_A_coords,idx)
        point_B_coords = self.get_points(self.point_B_coords,idx)
        
        # compute PCK reference length L_pck (equal to max bounding box side in image_A)
        N_pts = torch.sum(torch.ne(point_A_coords[0,:],-1))

        L_pck = torch.FloatTensor([torch.max(point_A_coords[:,:N_pts].max(1)[0]-point_A_coords[:,:N_pts].min(1)[0])])
                
        sample = {'source_image': image_A, 'target_image': image_B, 'source_im_size': im_size_A, 'target_im_size': im_size_B, 'source_points': point_A_coords, 'target_points': point_B_coords, 'L_pck': L_pck}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self,img_name_list,idx):
        img_name = os.path.join(self.dataset_path, img_name_list.iloc[idx])
        image = io.imread(img_name)
        
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
    
    def get_points(self,point_coords_list,idx):
        X=np.fromstring(point_coords_list.iloc[idx,0],sep=';')
        Y=np.fromstring(point_coords_list.iloc[idx,1],sep=';')
        Xpad = -np.ones(20); Xpad[:len(X)]=X
        Ypad = -np.ones(20); Ypad[:len(X)]=Y
        point_coords = np.concatenate((Xpad.reshape(1,20),Ypad.reshape(1,20)),axis=0)
        
        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords

    