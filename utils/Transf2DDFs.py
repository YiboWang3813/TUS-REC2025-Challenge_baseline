# This script contains two functions, which can be used to generate 4 DDFs, using transformations

import torch

def cal_global_ddfs(transformation_global,tform_calib_scale,image_points,landmark,w = 640,h = 480):
    """
    This function generates global DDF for all pixels and landmarks in a scan, using global transformations

    Args:
        transformation_global (torch.Tensor): shape=(N-1, 4, 4), global transformations for each frame in the scan; each transformation denotes the transformation from the current frame to the first frame
        tform_calib_scale (torch.Tensor): shape=(4, 4), scale from image coordinate system (in pixel) to image coordinate system (in mm)
        image_points (torch.Tensor): shape=(4, 307200), point coordinates for all pixels, in image coordinate system (in pixel) 
                                                        The coordinates defination of the 307200 points can be found in the function "reference_image_points" in utils/plot_functions.py
        landmark (torch.Tensor): shape=(100, 3), coordinates of landmarks in image coordinate system (in pixel).
                                                 The first dimension starts from 0, and the second and third dimensions start from 1, which is designed to be consistent with the calibration process.
        w (int): width of the image
        h (int): height of the image  

    Returns:
        global_allpts_DDF (numpy.ndarray): shape=(N-1, 3, 307200), global DDF for all pixels, where N-1 is the number of frames in that scan (excluding the first frame)
        global_landmark_DDF (numpy.ndarray): shape=(3, 100), global DDF for landmark 

    """
    # coordinates of points in current frame, with respect to the first frame 
    global_allpts = torch.matmul(transformation_global,torch.matmul(tform_calib_scale,image_points)) # (N-1, 4, 307200)
    # calculate DDF in mm, displacement from current frame to the first frame
    global_allpts_DDF = global_allpts[:,0:3,:]-torch.matmul(tform_calib_scale,image_points)[0:3,:].expand(global_allpts.shape[0],-1,-1) # (N-1, 3, 307200)
    # calculate DDF for landmark
    # As the coordinates of landmark start from 1, which is designed to be consistent with the calibration process, here we need to minus 1
    # 这里用landmark索引的时候还真是一次就能把3个轴都消掉只剩一个轴
    global_landmark_DDF = global_allpts_DDF.reshape(global_allpts_DDF.shape[0],-1,h,w)[landmark[:,0]-1,:,landmark[:,2]-1,landmark[:,1]-1].T.numpy() # (3, 100)
    global_allpts_DDF = global_allpts_DDF.numpy()

    return global_allpts_DDF,global_landmark_DDF


def cal_local_ddfs(transformation_local,tform_calib_scale,image_points,landmark,w = 640,h = 480):
    """
    This function generates local DDF for all pixels and landmarks in a scan, using local transformations

    Args:
        transformation_local (torch.Tensor): shape=(N-1, 4, 4), local transformations for each frame in the scan; each transformation denotes the transformation from the current frame to the previous frame
        tform_calib_scale (torch.Tensor): shape=(4, 4), scale from image coordinate system (in pixel) to image coordinate system (in mm) 
        image_points (torch.Tensor): shape=(4, 307200), point coordinates for all pixels, in image coordinate system (in pixel) 
                                                        The coordinates defination of the 307200 points can be found in the function "reference_image_points" in utils/plot_functions.py
        landmark (torch.Tensor): shape=(100, 3), coordinates of landmarks in image coordinate system (in pixel).
                                                 The first dimension starts from 0, and the second and third dimensions start from 1, which is designed to be consistent with the calibration process.
        w (int): width of the image
        h (int): height of the image 

    Returns:
        local_allpts_DDF (numpy.ndarray): shape=(N-1, 3, 307200), local DDF for all pixels, where N-1 is the number of frames in that scan (excluding the first frame)  
        local_landmark_DDF (numpy.ndarray): shape=(3, 100), local DDF for landmarks

    """

    # coordinates of points in current frame, with respect to the immediately previous frame 
    local_allpts = torch.matmul(transformation_local,torch.matmul(tform_calib_scale,image_points)) # (N-1, 4, 307200)
    # calculate DDF in mm, displacement from current frame to the immediately previous frame
    local_allpts_DDF = local_allpts[:,0:3,:]-torch.matmul(tform_calib_scale,image_points)[0:3,:].expand(local_allpts.shape[0],-1,-1) # (N-1, 3, 307200)
    # calculate DDF for landmark
    # As the coordinates of landmark start from 1, which is designed to be consistent with the calibration process, here we need to minus 1
    local_landmark_DDF = local_allpts_DDF.reshape(local_allpts_DDF.shape[0],-1,h,w)[landmark[:,0]-1,:,landmark[:,2]-1,landmark[:,1]-1].T.numpy() # (3, 100)
    local_allpts_DDF = local_allpts_DDF.numpy()
    
    return local_allpts_DDF,local_landmark_DDF