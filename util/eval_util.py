import torch
import numpy as np
import os
from skimage import draw
from geotnf.transformation import GeometricTnf,homography_mat_from_4_pts
from geotnf.point_tnf import compose_H_matrices, compose_aff_matrices, compose_tps
from geotnf.flow import th_sampling_grid_to_np_flow, write_flo_file
import torch.nn.functional as F
from data.pf_dataset import PFDataset, PFPascalDataset
from data.caltech_dataset import CaltechDataset
from torch.autograd import Variable
from geotnf.point_tnf import PointTnf, PointsToUnitCoords, PointsToPixelCoords
from util.py_util import create_file_path
from util.torch_util import expand_dim
from geotnf.point_tnf import PointTnf

def eval_model_multistage(model,geometric_model,num_of_iters,source_image,target_image):
    geoTnf = GeometricTnf(geometric_model=geometric_model, use_cuda=torch.cuda.is_available())
    for it in range(num_of_iters):
        # First iteration
        if it==0:
            theta = model({'source_image':source_image,'target_image':target_image})
            if geometric_model=='hom':
                theta = homography_mat_from_4_pts(theta)
            continue
        
        # Subsequent iterations

        # Compute warped image
        warped_image = None
        warped_image = geoTnf(source_image,theta)

        # Re-estimate tranformation
        theta_iter = model({'source_image':warped_image,'target_image':target_image})

        # update accumultated transformation
        if geometric_model=='hom':
            theta = compose_H_matrices(theta,homography_mat_from_4_pts(theta_iter))            
        elif geometric_model=='affine':
            theta = compose_aff_matrices(theta,theta_iter)
        elif geometric_model=='tps':
            theta = compose_tps(theta,theta_iter)

    # warp one last time using final transformation
    warped_image = None
    warped_image = geoTnf(source_image,theta)

    return (theta,warped_image)

def compute_metric(metric,model_1,geometric_model_1,model_2,geometric_model_2,dataset,dataloader,batch_tnf,batch_size,args=None):
    # Initialize stats
    two_stage=(model_2 is not None)
    N=len(dataset)
    stats={}
    # decide which results should be computed aff/tps/aff+tps
    stats[geometric_model_1]={}
    if two_stage:
        stats[geometric_model_1+'_'+geometric_model_2]={}
    
    # choose metric function and metrics to compute
    if metric=='pck':  
        metrics = ['pck']
        metric_fun = pck_metric
    if metric=='dist':
        metrics = ['dist']
        metric_fun = point_dist_metric
    elif metric=='area':
        metrics = ['intersection_over_union',
                   'label_transfer_accuracy',
                   'localization_error']
        metric_fun = area_metrics
    elif metric=='flow':
        metrics = ['flow']
        metric_fun = flow_metrics
    # initialize vector for storing results for each metric
    for key in stats.keys():
        for metric in metrics:
            stats[key][metric] = np.zeros((N,1))

    # Compute
    for i, batch in enumerate(dataloader):
        batch = batch_tnf(batch)        
        batch_start_idx=batch_size*i

        model_1.eval()
        theta_1=None
        theta_2=None

        source_image = batch['source_image']
        target_image = batch['target_image']

        # compute iterative first stage       
        theta_1,warped_image_1 = eval_model_multistage(model_1,geometric_model_1,args.num_of_iters,source_image,target_image)

        #import pdb;pdb.set_trace()

        # compute single second stage
        if two_stage:
            model_2.eval()
            theta_2 = model_2({'source_image':warped_image_1,'target_image':target_image})           
    
        if metric_fun is not None:
            stats = metric_fun(batch,batch_start_idx,theta_1,theta_2,geometric_model_1,geometric_model_2,stats,args)
            
        print('Batch: [{}/{} ({:.0f}%)]'.format(i, len(dataloader), 100. * i / len(dataloader)))

    if metric=='flow':
        print('Flow results have been saved to :'+args.flow_output_dir)
        return stats

    # Print results
    for key in stats.keys():
        print('=== Results '+key+' ===')
        for metric in metrics:
            results=stats[key][metric]
            good_idx = np.flatnonzero((results!=-1) * ~np.isnan(results))
            print('Total: '+str(results.size))
            print('Valid: '+str(good_idx.size)) 
            filtered_results = results[good_idx]
            print(metric+':','{:.2%}'.format(np.mean(filtered_results)))
                
        print('\n')
        
    return stats


def pck(source_points,warped_points,L_pck,alpha=0.1):
    # compute precentage of correct keypoints
    batch_size=source_points.size(0)
    pck=torch.zeros((batch_size))
    for i in range(batch_size):
        p_src = source_points[i,:]
        p_wrp = warped_points[i,:]
        N_pts = torch.sum(torch.ne(p_src[0,:],-1)*torch.ne(p_src[1,:],-1))
        point_distance = torch.pow(torch.sum(torch.pow(p_src[:,:N_pts]-p_wrp[:,:N_pts],2),0),0.5)
        L_pck_mat = L_pck[i].expand_as(point_distance)
        correct_points = torch.le(point_distance,L_pck_mat*alpha)
        pck[i]=torch.mean(correct_points.float())
    return pck

def pck_metric(batch,batch_start_idx,theta_1,theta_2,geometric_model_1,geometric_model_2,stats,args,use_cuda=True):

    two_stage=(geometric_model_2 is not None)

    alpha = args.pck_alpha
       
    source_im_size = batch['source_im_size']
    target_im_size = batch['target_im_size']

    source_points = batch['source_points']
    target_points = batch['target_points']
    
    # Instantiate point transformer
    pt = PointTnf(use_cuda=use_cuda,
                  tps_reg_factor=args.tps_reg_factor)

    if geometric_model_1=='affine':
        tnf_1 = pt.affPointTnf
    elif geometric_model_1=='hom':
        tnf_1 = pt.homPointTnf
    elif geometric_model_1=='tps':
        tnf_1 = pt.tpsPointTnf

    if two_stage:
        if geometric_model_2=='affine':
            tnf_2 = pt.affPointTnf
        elif geometric_model_2=='hom':
            tnf_2 = pt.homPointTnf
        elif geometric_model_2=='tps':
            tnf_2 = pt.tpsPointTnf        

    # warp points with estimated transformations
    target_points_norm = PointsToUnitCoords(target_points,target_im_size)

    # compute points stage 1 only
    warped_points_1_norm = tnf_1(theta_1,target_points_norm)
    warped_points_1 = PointsToPixelCoords(warped_points_1_norm,source_im_size)

    if two_stage:        
        # do tps+affine
        warped_points_1_2_norm = tnf_2(theta_2,target_points_norm)
        warped_points_1_2_norm = tnf_1(theta_1,warped_points_1_2_norm)
        warped_points_1_2 = PointsToPixelCoords(warped_points_1_2_norm,source_im_size)
    

    L_pck = batch['L_pck'].data
    
    current_batch_size=batch['source_im_size'].size(0)
    indices = range(batch_start_idx,batch_start_idx+current_batch_size)

    # compute PCK
    pck_1 = pck(source_points.data, warped_points_1.data, L_pck, alpha)
    stats[geometric_model_1]['pck'][indices] = pck_1.unsqueeze(1).cpu().numpy()
    if two_stage:
        pck_1_2 = pck(source_points.data, warped_points_1_2.data, L_pck, alpha)
        stats[geometric_model_1+'_'+geometric_model_2]['pck'][indices] = pck_1_2.unsqueeze(1).cpu().numpy() 
        
    return stats

def area_metrics(batch,batch_start_idx,theta_1,theta_2,geometric_model_1,geometric_model_2,stats,args,use_cuda=True):
    two_stage=(geometric_model_2 is not None)

    pt=PointTnf(use_cuda=use_cuda)
    
    if geometric_model_1=='affine':
        tnf_1 = pt.affPointTnf
    elif geometric_model_1=='hom':
        tnf_1 = pt.homPointTnf
    elif geometric_model_1=='tps':
        tnf_1 = pt.tpsPointTnf

    if two_stage:
        if geometric_model_2=='affine':
            tnf_2 = pt.affPointTnf
        elif geometric_model_2=='hom':
            tnf_2 = pt.homPointTnf
        elif geometric_model_2=='tps':
            tnf_2 = pt.tpsPointTnf        

    batch_size=batch['source_im_size'].size(0)
    for b in range(batch_size):
        h_src = int(batch['source_im_size'][b,0].data.cpu().numpy())
        w_src = int(batch['source_im_size'][b,1].data.cpu().numpy())
        h_tgt = int(batch['target_im_size'][b,0].data.cpu().numpy())
        w_tgt = int(batch['target_im_size'][b,1].data.cpu().numpy())

        target_mask_np,target_mask = poly_str_to_mask(batch['target_polygon'][0][b],
                                                      batch['target_polygon'][1][b],
                                                      h_tgt,w_tgt,use_cuda=use_cuda)

        source_mask_np,source_mask = poly_str_to_mask(batch['source_polygon'][0][b],
                                                      batch['source_polygon'][1][b],
                                                      h_src,w_src,use_cuda=use_cuda)

        grid_X,grid_Y = np.meshgrid(np.linspace(-1,1,w_tgt),np.linspace(-1,1,h_tgt))
        grid_X = torch.FloatTensor(grid_X).unsqueeze(0).unsqueeze(3)
        grid_Y = torch.FloatTensor(grid_Y).unsqueeze(0).unsqueeze(3)
        grid_X = Variable(grid_X,requires_grad=False)
        grid_Y = Variable(grid_Y,requires_grad=False)
        if use_cuda:
            grid_X = grid_X.cuda()
            grid_Y = grid_Y.cuda()

        grid_X_vec = grid_X.view(1,1,-1)
        grid_Y_vec = grid_Y.view(1,1,-1)

        grid_XY_vec = torch.cat((grid_X_vec,grid_Y_vec),1)        

        def pointsToGrid (x,h_tgt=h_tgt,w_tgt=w_tgt): return x.contiguous().view(1,2,h_tgt,w_tgt).transpose(1,2).transpose(2,3)

        idx = batch_start_idx+b
        
        # stage 1
        grid_1 = pointsToGrid(tnf_1(theta_1[b,:].unsqueeze(0),grid_XY_vec))
        warped_mask_1 = F.grid_sample(source_mask, grid_1)            
        flow_1 = th_sampling_grid_to_np_flow(source_grid=grid_1,h_src=h_src,w_src=w_src)
        
        stats[geometric_model_1]['intersection_over_union'][idx] = intersection_over_union(warped_mask_1,target_mask).cpu().numpy()
        stats[geometric_model_1]['label_transfer_accuracy'][idx] = label_transfer_accuracy(warped_mask_1,target_mask).cpu().numpy()
        stats[geometric_model_1]['localization_error'][idx] = localization_error(source_mask_np, target_mask_np, flow_1)
        
        if two_stage:
            grid_1_2 = pointsToGrid(tnf_1(theta_1[b,:].unsqueeze(0),tnf_2(theta_2[b,:].unsqueeze(0),grid_XY_vec)))
            warped_mask_1_2 = F.grid_sample(source_mask, grid_1_2)
            flow_1_2 = th_sampling_grid_to_np_flow(source_grid=grid_1_2,h_src=h_src,w_src=w_src)
            
            stats[geometric_model_1+'_'+geometric_model_2]['intersection_over_union'][idx] = intersection_over_union(warped_mask_1_2,target_mask).cpu().numpy()
            stats[geometric_model_1+'_'+geometric_model_2]['label_transfer_accuracy'][idx] = label_transfer_accuracy(warped_mask_1_2,target_mask).cpu().numpy()
            stats[geometric_model_1+'_'+geometric_model_2]['localization_error'][idx] = localization_error(source_mask_np, target_mask_np, flow_1_2)        

    return stats


def flow_metrics(batch,batch_start_idx,theta_1,theta_2,geometric_model_1,geometric_model_2,stats,args,use_cuda=True):
    two_stage=(geometric_model_2 is not None)

    result_path=args.flow_output_dir

    pt=PointTnf(use_cuda=use_cuda)

    if geometric_model_1=='affine':
        tnf_1 = pt.affPointTnf
    elif geometric_model_1=='hom':
        tnf_1 = pt.homPointTnf
    elif geometric_model_1=='tps':
        tnf_1 = pt.tpsPointTnf

    if two_stage:
        if geometric_model_2=='affine':
            tnf_2 = pt.affPointTnf
        elif geometric_model_2=='hom':
            tnf_2 = pt.homPointTnf
        elif geometric_model_2=='tps':
            tnf_2 = pt.tpsPointTnf    
    
    batch_size=batch['source_im_size'].size(0)
    for b in range(batch_size):
        h_src = int(batch['source_im_size'][b,0].data.cpu().numpy())
        w_src = int(batch['source_im_size'][b,1].data.cpu().numpy())
        h_tgt = int(batch['target_im_size'][b,0].data.cpu().numpy())
        w_tgt = int(batch['target_im_size'][b,1].data.cpu().numpy())

        grid_X,grid_Y = np.meshgrid(np.linspace(-1,1,w_tgt),np.linspace(-1,1,h_tgt))
        grid_X = torch.FloatTensor(grid_X).unsqueeze(0).unsqueeze(3)
        grid_Y = torch.FloatTensor(grid_Y).unsqueeze(0).unsqueeze(3)
        grid_X = Variable(grid_X,requires_grad=False)
        grid_Y = Variable(grid_Y,requires_grad=False)
        if use_cuda:
            grid_X = grid_X.cuda()
            grid_Y = grid_Y.cuda()

        grid_X_vec = grid_X.view(1,1,-1)
        grid_Y_vec = grid_Y.view(1,1,-1)

        grid_XY_vec = torch.cat((grid_X_vec,grid_Y_vec),1)        

        def pointsToGrid (x,h_tgt=h_tgt,w_tgt=w_tgt): return x.contiguous().view(1,2,h_tgt,w_tgt).transpose(1,2).transpose(2,3)
                
        grid_1 = pointsToGrid(tnf_1(theta_1[b,:].unsqueeze(0),grid_XY_vec))
        flow_1 = th_sampling_grid_to_np_flow(source_grid=grid_1,h_src=h_src,w_src=w_src)
        flow_1_path = os.path.join(result_path,geometric_model_1,batch['flow_path'][b])
        create_file_path(flow_1_path)
        write_flo_file(flow_1,flow_1_path)

        if two_stage:
            grid_1_2 = pointsToGrid(tnf_1(theta_1[b,:].unsqueeze(0),tnf_2(theta_2[b,:].unsqueeze(0),grid_XY_vec)))
            flow_1_2 = th_sampling_grid_to_np_flow(source_grid=grid_1_2,h_src=h_src,w_src=w_src)
            flow_1_2_path = os.path.join(result_path,geometric_model_1+'_'+geometric_model_2,batch['flow_path'][b])
            create_file_path(flow_1_2_path)
            write_flo_file(flow_1_2,flow_1_2_path)

    return stats

def poly_to_mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def poly_str_to_mask(poly_x_str,poly_y_str,out_h,out_w,use_cuda=True):
    polygon_x = np.fromstring(poly_x_str,sep=',')
    polygon_y = np.fromstring(poly_y_str,sep=',')
    mask_np = poly_to_mask(vertex_col_coords=polygon_x,
                               vertex_row_coords=polygon_y,shape=[out_h,out_w])
    mask = Variable(torch.FloatTensor(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0))
    if use_cuda:
        mask = mask.cuda()
    return (mask_np,mask)

def intersection_over_union(warped_mask,target_mask): 
    relative_part_weight = torch.sum(torch.sum(target_mask.data.gt(0.5).float(),2,True),3,True)/torch.sum(target_mask.data.gt(0.5).float())
    part_iou = torch.sum(torch.sum((warped_mask.data.gt(0.5) & target_mask.data.gt(0.5)).float(),2,True),3,True)/torch.sum(torch.sum((warped_mask.data.gt(0.5) | target_mask.data.gt(0.5)).float(),2,True),3,True)
    weighted_iou = torch.sum(torch.mul(relative_part_weight,part_iou))
    return weighted_iou

def label_transfer_accuracy(warped_mask,target_mask): 
    return torch.mean((warped_mask.data.gt(0.5) == target_mask.data.gt(0.5)).double())

def localization_error(source_mask_np, target_mask_np, flow_np):
    h_tgt, w_tgt = target_mask_np.shape[0],target_mask_np.shape[1]
    h_src, w_src = source_mask_np.shape[0],source_mask_np.shape[1]

    # initial pixel positions x1,y1 in target image
    x1, y1 = np.meshgrid(range(1,w_tgt+1), range(1,h_tgt+1))
    # sampling pixel positions x2,y2
    x2 = x1 + flow_np[:,:,0]
    y2 = y1 + flow_np[:,:,1]

    # compute in-bound coords for each image
    in_bound = (x2 >= 1) & (x2 <= w_src) & (y2 >= 1) & (y2 <= h_src)
    row,col = np.where(in_bound)
    row_1=y1[row,col].flatten().astype(np.int)-1
    col_1=x1[row,col].flatten().astype(np.int)-1
    row_2=y2[row,col].flatten().astype(np.int)-1
    col_2=x2[row,col].flatten().astype(np.int)-1

    # compute relative positions
    target_loc_x,target_loc_y = obj_ptr(target_mask_np)
    source_loc_x,source_loc_y = obj_ptr(source_mask_np)
    x1_rel=target_loc_x[row_1,col_1]
    y1_rel=target_loc_y[row_1,col_1]
    x2_rel=source_loc_x[row_2,col_2]
    y2_rel=source_loc_y[row_2,col_2]

    # compute localization error
    loc_err = np.mean(np.abs(x1_rel-x2_rel)+np.abs(y1_rel-y2_rel))
    
    return loc_err

def obj_ptr(mask):
    # computes images of normalized coordinates around bounding box
    # kept function name from DSP code
    h,w = mask.shape[0],mask.shape[1]
    y, x = np.where(mask>0.5)
    left = np.min(x);
    right = np.max(x);
    top = np.min(y);
    bottom = np.max(y);
    fg_width = right-left + 1;
    fg_height = bottom-top + 1;
    x_image,y_image = np.meshgrid(range(1,w+1), range(1,h+1));
    x_image = (x_image - left)/fg_width;
    y_image = (y_image - top)/fg_height;
    return (x_image,y_image)
