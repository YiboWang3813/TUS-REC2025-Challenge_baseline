# this is the function for generating DDF
import os
import json
from utils.Prediction import Prediction
from utils.funs import load_config

def predict_ddfs(frames,landmark,data_path_calib,device):
    """
    Args:
        frames (numpy.ndarray): shape=(N, 480, 640),frames in the scan, where N is the number of frames in this scan
        landmark (numpy.ndarray): shape=(20,3), denoting the location of landmark. For example, a landmark with location of (10,200,100) denotes the landmark on the 10th frame, with coordinate of (200,100)
        data_path_calib (str): path to calibration matrix
        device (str): device to use for prediction, either 'cuda' or 'cpu'

    Returns:
        pred_global_allpts_DDF (numpy.ndarray): shape=(N-1, 3, 307200), global DDF for all pixels, where N-1 is the number of frames in that scan (excluding the first frame)
        pred_global_landmark_DDF (numpy.ndarray): shape=(3, 100), global DDF for landmark  
        pred_local_allpts_DDF (numpy.ndarray): shape=(N-1, 3, 307200), local DDF for all pixels, where N-1 is the number of frames in that scan (excluding the first frame) 
        pred_local_landmark_DDF (numpy.ndarray): shape=(3, 100), local DDF for landmark
    
    """
    
    # path to the trained baseline model
    model_path = os.getcwd() + '/results/seq_len2__efficientnet_b1__lr0.0001__pred_type_parameter__label_type_point'
    model_name = 'saved_model/best_validation_dist_model'
    # parameters used in baseline code
    _,parameters = load_config(model_path+'/'+'config.txt')

    prediction = Prediction(parameters,model_name,data_path_calib,model_path,device)
    # generate 4 DDFs for the scan
    pred_global_allpts_DDF,\
    pred_global_landmark_DDF,\
    pred_local_allpts_DDF,\
    pred_local_landmark_DDF = prediction.generate_prediction_DDF(frames,landmark)

    return pred_global_allpts_DDF,pred_global_landmark_DDF,pred_local_allpts_DDF,pred_local_landmark_DDF

