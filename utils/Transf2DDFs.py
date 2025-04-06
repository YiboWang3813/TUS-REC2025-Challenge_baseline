# This script contains four functions, which can be used to generate required DDFs, using transformations

import torch
import numpy as np


def cal_global_ddfs1(transformation_global,tform_calib_scale,image_points,landmark,w = 640,h = 480):
    """
    This function generates global DDF for all pixels and landmarks in a scan, using global transformations

    Args:
        transformation_global (torch.Tensor): shape=(N-1, 4, 4), global transformations for each frame in the scan; each transformation denotes the transformation from the current frame to the first frame
        tform_calib_scale (torch.Tensor): shape=(4, 4), scale from image coordinate system (in pixel) to image coordinate system (in mm)
        image_points (torch.Tensor): shape=(4, 4), point coordinate for four corner pixels, in image coordinate system (in pixel) 
        landmark (torch.Tensor): shape=(100, 3), coordinates of landmark points in image coordinate system (in pixel)
        w (int): width of the image
        h (int): height of the image      
    Returns:
        global_allpts_DDF (numpy.ndarray): shape=(N-1, 3, 307200), global DDF for all pixels, where N-1 is the number of frames in that scan (excluding the first frame)
        global_landmark_DDF (numpy.ndarray): shape=(3, 20), global DDF for landmark 
    """
    # coordinates of points in current frame, with respect to the first frame 
    global_corner_pts = torch.matmul(transformation_global,torch.matmul(tform_calib_scale,image_points))
    # calculate DDF in mm, displacement from current frame to the first frame
    global_corner_pts_DDF = global_corner_pts[:,0:3,:]-torch.matmul(tform_calib_scale,image_points)[0:3,:].expand(global_corner_pts.shape[0],-1,-1)
    global_corner_pts_DDF = global_corner_pts_DDF.cpu()

    # interpolation
    i,j = torch.meshgrid(torch.arange(0,h),torch.arange(0,w),indexing='ij')
    d_long = (global_corner_pts_DDF[:,:,1] - global_corner_pts_DDF[:,:,0])/(w-1)
    d_short = (global_corner_pts_DDF[:,:,2] - global_corner_pts_DDF[:,:,0])/(h-1)
    global_allpts_DDF = global_corner_pts_DDF[:,:,0,None,None] + d_short[:,:,None,None]*i + d_long[:,:,None,None]*j
    # calculate DDF for landmark
    global_landmark_DDF = global_allpts_DDF[landmark[:,0]-1,:,landmark[:,2],landmark[:,1]].T.numpy()
    # reshape global_allpts_DDF to (N-1, 3, 307200)
    global_allpts_DDF = global_allpts_DDF.reshape(global_allpts_DDF.shape[0],3,-1).numpy()

    return global_allpts_DDF,global_landmark_DDF

def cal_local_ddfs1(transformation_local,tform_calib_scale,image_points,landmark,w = 640,h = 480):
    """
    This function generates local DDF for all pixels in a scan, using local transformations

    Args:
        transformation_local (torch.Tensor): shape=(N-1, 4, 4), local transformations for each frame in the scan; each transformation denotes the transformation from the current frame to the previous frame
        tform_calib_scale (torch.Tensor): shape=(4, 4), scale from image coordinate system (in pixel) to image coordinate system (in mm) 
        image_points (torch.Tensor): shape=(4, 4), point coordinate for four corner pixels, in image coordinate system (in pixel) 
        landmark (torch.Tensor): shape=(100, 3), coordinates of landmark points in image coordinate system (in pixel)
        w (int): width of the image
        h (int): height of the image   
    Returns:
        local_allpts_DDF (numpy.ndarray): shape=(N-1, 3, 307200), local DDF for all pixels, where N-1 is the number of frames in that scan (excluding the first frame)  
        local_landmark_DDF (numpy.ndarray): shape=(3, 20), local DDF for landmarks

    """

    # coordinates of points in current frame, with respect to the immediately previous frame 
    local_corner_pts = torch.matmul(transformation_local,torch.matmul(tform_calib_scale,image_points))
    # calculate DDF in mm, displacement from current frame to the immediately previous frame
    local_corner_pts_DDF = local_corner_pts[:,0:3,:]-torch.matmul(tform_calib_scale,image_points)[0:3,:].expand(local_corner_pts.shape[0],-1,-1)
    local_corner_pts_DDF = local_corner_pts_DDF.cpu().numpy()

    # interpolation
    i,j = np.meshgrid(np.arange(0,h),np.arange(0,w),indexing='ij')
    d_long = (local_corner_pts_DDF[:,:,1] - local_corner_pts_DDF[:,:,0])/(w-1)
    d_short = (local_corner_pts_DDF[:,:,2] - local_corner_pts_DDF[:,:,0])/(h-1)
    local_allpts_DDF = local_corner_pts_DDF[:,:,0,None,None] + d_short[:,:,None,None]*i + d_long[:,:,None,None]*j
    # calculate DDF for landmark
    local_landmark_DDF = np.transpose(local_allpts_DDF[landmark[:,0]-1,:,landmark[:,2],landmark[:,1]])
    # reshape local_allpts_DDF to (N-1, 3, 307200)
    local_allpts_DDF = local_allpts_DDF.reshape(local_allpts_DDF.shape[0],3,-1)

    return local_allpts_DDF,local_landmark_DDF



def cal_global_ddfs(transformation_global,tform_calib_scale,image_points,landmark,w = 640,h = 480):
    """
    This function generates global DDF for all pixels in a scan, using global transformations

    Args:
        transformation_global (torch.Tensor): shape=(N-1, 4, 4), global transformations for each frame in the scan; each transformation denotes the transformation from the current frame to the first frame
        tform_calib_scale (torch.Tensor): shape=(4, 4), scale from image coordinate system (in pixel) to image coordinate system (in mm)
        image_points (torch.Tensor): shape=(4, 307200), point coordinate for all pixels, in image coordinate system (in pixel) 
    
    Returns:
        global_allpts_DDF (numpy.ndarray): shape=(N-1, 3, 307200), global DDF for all pixels, where N-1 is the number of frames in that scan (excluding the first frame)

    """
    # coordinates of points in current frame, with respect to the first frame 
    global_allpts = torch.matmul(transformation_global,torch.matmul(tform_calib_scale,image_points))
    # calculate DDF in mm, displacement from current frame to the first frame
    global_allpts_DDF = global_allpts[:,0:3,:]-torch.matmul(tform_calib_scale,image_points)[0:3,:].expand(global_allpts.shape[0],-1,-1)
    # calculate DDF for landmark
    global_landmark_DDF = global_allpts_DDF.reshape(global_allpts_DDF.shape[0],-1,h,w)[landmark[:,0]-1,:,landmark[:,2],landmark[:,1]].T.cpu().numpy()
    global_allpts_DDF = global_allpts_DDF.cpu().numpy()

    return global_allpts_DDF,global_landmark_DDF

def cal_global_landmark(transformation_global,landmark,tform_calib_scale):
    """
    This function generates global DDF for landmark, using global transformations

    Args:
        transformation_global (torch.Tensor): shape=(N-1, 4, 4), global transformations for each frame in the scan; each transformation denotes the transformation from the current frame to the first frame, where N-1 is the number of frames in that scan (excluding the first frame)
        landmark (torch.Tensor): shape=(20, 3), coordinates of landmark points in image coordinate system (in pixel)
        tform_calib_scale (torch.Tensor): shape=(4, 4), scale from image coordinate system (in pixel) to image coordinate system (in mm) 

    Returns:
       global_landmark_DDF (numpy.ndarray): shape=(3, 20), global DDF for landmark  
    """
    
    global_landmark = torch.zeros(3,len(landmark))
    for i in range(len(landmark)):  
        # point coordinate in image coordinate system (in pixel)  
        pts_coord = torch.cat((landmark[i][1:]+1, torch.FloatTensor([0,1])),axis = 0).cuda()
        # calculate global DDF in mm, displacement from current frame to the first frame
        global_landmark[:,i] = torch.matmul(transformation_global[landmark[i][0]-1],torch.matmul(tform_calib_scale,pts_coord))[0:3]-torch.matmul(tform_calib_scale,pts_coord)[0:3]

    global_landmark_DDF = global_landmark.numpy()

    return global_landmark_DDF

def cal_local_ddfs(transformation_local,tform_calib_scale,image_points,landmark,w = 640,h = 480):
    """
    This function generates local DDF for all pixels in a scan, using local transformations

    Args:
        transformation_local (torch.Tensor): shape=(N-1, 4, 4), local transformations for each frame in the scan; each transformation denotes the transformation from the current frame to the previous frame
        tform_calib_scale (torch.Tensor): shape=(4, 4), scale from image coordinate system (in pixel) to image coordinate system (in mm) 
        image_points (torch.Tensor): shape=(4, 307200), point coordinate for all pixels, in image coordinate system (in pixel) 
    
    Returns:
        local_allpts_DDF (numpy.ndarray): shape=(N-1, 3, 307200), local DDF for all pixels, where N-1 is the number of frames in that scan (excluding the first frame)  
    """

    # coordinates of points in current frame, with respect to the immediately previous frame 
    local_allpts = torch.matmul(transformation_local,torch.matmul(tform_calib_scale,image_points))
    # calculate DDF in mm, displacement from current frame to the immediately previous frame
    local_allpts_DDF = local_allpts[:,0:3,:]-torch.matmul(tform_calib_scale,image_points)[0:3,:].expand(local_allpts.shape[0],-1,-1)
    # calculate DDF for landmark
    local_landmark_DDF = local_allpts_DDF.reshape(local_allpts_DDF.shape[0],-1,h,w)[landmark[:,0]-1,:,landmark[:,2],landmark[:,1]].T.cpu().numpy()
    local_allpts_DDF = local_allpts_DDF.cpu().numpy()
    
    return local_allpts_DDF,local_landmark_DDF

def cal_local_landmark(transformation_local,landmark,tform_calib_scale):
    """
    This function generates local DDF for landmark in a scan, using local transformations

    Args:
        transformation_local (torch.Tensor): shape=(N-1, 4, 4), local transformations for each frame in the scan; each transformation denotes the transformation from the current frame to the previous frame, where N-1 is the number of frames in that scan (excluding the first frame)
        landmark (torch.Tensor): shape=(20, 3), coordinates of landmark points in image coordinate system (in pixel)
        tform_calib_scale (torch.Tensor): shape=(4, 4), scale from image coordinate system (in pixel) to image coordinate system (in mm)

    Returns:
        local_landmark_DDF (numpy.ndarray): shape=(3, 20), local DDF for landmarks
    """

    local_landmark = torch.zeros(3,len(landmark))
    for i in range(len(landmark)):  
        # point coordinate in image coordinate system (in pixel)  
        pts_coord = torch.cat((landmark[i][1:]+1, torch.FloatTensor([0,1])),axis = 0).cuda()
        # calculate DDF in mm, displacement from current frame to the immediately previous frame
        local_landmark[:,i] = torch.matmul(transformation_local[landmark[i][0]-1],torch.matmul(tform_calib_scale,pts_coord))[0:3]-torch.matmul(tform_calib_scale,pts_coord)[0:3]

    local_landmark_DDF = local_landmark.numpy()

    return local_landmark_DDF