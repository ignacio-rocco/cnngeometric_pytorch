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
        training_image_path (string): Directory with the images.
        output_size (2-tuple): Desired output size
        transform (callable): Transformation for post-processing the training pair (eg. image normalization)
        
    """

    def __init__(self, csv_file, training_image_path, output_size=(240, 240), transform=None):
        self.out_h, self.out_w = output_size
        self.train_data = pd.read_csv(csv_file)
        self.img_A_names = self.train_data.iloc[:, 0]
        self.img_B_names = self.train_data.iloc[:, 1]
        self.point_A_coords = self.train_data.iloc[:, 2:22].values.astype('float')
        self.point_B_coords = self.train_data.iloc[:, 22:].values.astype('float')
        self.training_image_path = training_image_path
        self.transform = transform
        # no cuda as dataset is called from CPU threads in dataloader and produces confilct
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda=False)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # get pre-processed images
        image_A, im_size_A = self.get_image(self.img_A_names, idx)
        image_B, im_size_B = self.get_image(self.img_B_names, idx)

        # get pre-processed point coords
        point_A_coords = self.get_points(self.point_A_coords, idx)
        point_B_coords = self.get_points(self.point_B_coords, idx)

        # compute PCK reference length L_pck (equal to max bounding box side in image_A)
        L_pck = torch.FloatTensor([torch.max(point_A_coords.max(1)[0] - point_A_coords.min(1)[0])])

        sample = {'source_image': image_A, 'target_image': image_B, 'source_im_size': im_size_A,
                  'target_im_size': im_size_B, 'source_points': point_A_coords, 'target_points': point_B_coords,
                  'L_pck': L_pck}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self, img_name_list, idx):
        img_name = os.path.join(self.training_image_path, img_name_list[idx])
        image = io.imread(img_name)

        # get image size
        im_size = np.asarray(image.shape)

        # convert to torch Variable
        image = np.expand_dims(image.transpose((2, 0, 1)), 0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image, requires_grad=False)

        # Resize image using bilinear sampling with identity affine tnf
        image = self.affineTnf(image_var).data.squeeze(0)

        im_size = torch.Tensor(im_size.astype(np.float32))

        return (image, im_size)

    def get_points(self, point_coords_list, idx):
        point_coords = point_coords_list[idx, :].reshape(2, 10)

        #        # swap X,Y coords, as the the row,col order (Y,X) is used for computations
        #        point_coords = point_coords[[1,0],:]

        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords
