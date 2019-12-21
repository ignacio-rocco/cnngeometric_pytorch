import torch
from torch.autograd import Variable
import numpy as np
from geotnf.grid_gen import TpsGridGen
from geotnf.transformation import homography_mat_from_4_pts

def normalize_axis(x,L):
    return (x-1-(L-1)/2)*2/(L-1)

def unnormalize_axis(x,L):
    return x*(L-1)/2+1+(L-1)/2
    
class PointTnf(object):
    """
    
    Class with functions for transforming a set of points with affine/tps transformations
    
    """
    def __init__(self, tps_grid_size=3, tps_reg_factor=0, use_cuda=True):
        self.use_cuda=use_cuda
        self.tpsTnf = TpsGridGen(grid_size=tps_grid_size,
                                 reg_factor=tps_reg_factor,
                                 use_cuda=self.use_cuda)   
    

    def tpsPointTnf(self,theta,points):
        # points are expected in [B,2,N], where first row is X and second row is Y
        # reshape points for applying Tps transformation
        points=points.unsqueeze(3).transpose(1,3)
        # apply transformation
        warped_points = self.tpsTnf.apply_transformation(theta,points)
        # undo reshaping
        warped_points=warped_points.transpose(3,1).squeeze(3)      
        return warped_points

    def homPointTnf(self,theta,points,eps=1e-5):
        b=theta.size(0)
        if theta.size(1)==9:
            H = theta            
        else:
            H = homography_mat_from_4_pts(theta)            
        h0=H[:,0].unsqueeze(1).unsqueeze(2)
        h1=H[:,1].unsqueeze(1).unsqueeze(2)
        h2=H[:,2].unsqueeze(1).unsqueeze(2)
        h3=H[:,3].unsqueeze(1).unsqueeze(2)
        h4=H[:,4].unsqueeze(1).unsqueeze(2)
        h5=H[:,5].unsqueeze(1).unsqueeze(2)
        h6=H[:,6].unsqueeze(1).unsqueeze(2)
        h7=H[:,7].unsqueeze(1).unsqueeze(2)
        h8=H[:,8].unsqueeze(1).unsqueeze(2)

        X=points[:,0,:].unsqueeze(1)
        Y=points[:,1,:].unsqueeze(1)
        Xp = X*h0+Y*h1+h2
        Yp = X*h3+Y*h4+h5
        k = X*h6+Y*h7+h8
        # prevent division by 0
        k = k+torch.sign(k)*eps

        Xp /= k; Yp /= k

        return torch.cat((Xp,Yp),1)
    
    def affPointTnf(self,theta,points):
        theta_mat = theta.view(-1,2,3)
        warped_points = torch.bmm(theta_mat[:,:,:2],points)
        warped_points += theta_mat[:,:,2].unsqueeze(2).expand_as(warped_points)
        return warped_points

def PointsToUnitCoords(P,im_size):
    h,w = im_size[:,0],im_size[:,1]
    P_norm = P.clone()
    # normalize Y
    P_norm[:,0,:] = normalize_axis(P[:,0,:],w.unsqueeze(1).expand_as(P[:,0,:]))
    # normalize X
    P_norm[:,1,:] = normalize_axis(P[:,1,:],h.unsqueeze(1).expand_as(P[:,1,:]))
    return P_norm

def PointsToPixelCoords(P,im_size):
    h,w = im_size[:,0],im_size[:,1]
    P_norm = P.clone()
    # normalize Y
    P_norm[:,0,:] = unnormalize_axis(P[:,0,:],w.unsqueeze(1).expand_as(P[:,0,:]))
    # normalize X
    P_norm[:,1,:] = unnormalize_axis(P[:,1,:],h.unsqueeze(1).expand_as(P[:,1,:]))
    return P_norm

def compose_H_matrices(H1,H2):
    H1=H1.contiguous().view(-1,3,3)
    H2=H2.contiguous().view(-1,3,3)
    H = torch.bmm(H1,H2).view(-1,9)
    return H
    
def compose_aff_matrices(theta_1,theta_2):
    batch_size=theta_1.size()[0]
    O=torch.zeros((batch_size,1,3)); O[:,:,2]=1
    O=Variable(O)
    if theta_1.is_cuda:
        O=O.cuda()
    theta_1=torch.cat((theta_1.contiguous().view(-1,2,3),O),1)
    theta_2=torch.cat((theta_2.contiguous().view(-1,2,3),O),1)

    theta = torch.bmm(theta_1,theta_2)[:,:2,:].contiguous().view(batch_size,6)
    return theta

def compose_tps(theta_1,theta_2):
    batch_size=theta_1.size()[0]
    use_cuda=torch.cuda.is_available()
    pt = PointTnf(use_cuda=use_cuda)
    
    P_1_2 = pt.tpsPointTnf(theta=theta_1.contiguous(),points=theta_2.view(batch_size,2,9))
    
    theta = P_1_2.contiguous().view(-1,18)
    return theta