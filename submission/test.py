# This script is an example of the test process

import os,h5py
import numpy as np
import time
import torch
from utils.metrics import cal_dist
from predict_ddfs import predict_ddfs
from utils.generate_ddf_from_label import generate_ddf_from_label
from utils.plot_scans import plot_scans

def main():
    # folder name in docker container that external folders will be mounted to
    DATA_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data')
    RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)),'results')

    # path to data folders
    data_path_frame = os.path.join(DATA_FOLDER,'frames')
    data_path_transf = os.path.join(DATA_FOLDER,'transfs')
    data_path_landmark = os.path.join(DATA_FOLDER,'landmarks')
    data_path_calib = os.path.join(DATA_FOLDER,'calib_matrix.csv')
    data_path_metric = RESULTS_FOLDER
    data_path_plot = os.path.join(RESULTS_FOLDER,'plot')
    # keys for all the scans in the dataset
    dataset_keys = h5py.File(os.path.join(DATA_FOLDER,"dataset_keys.h5"),'r')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # for ground truth use
    generate_GT_ddf = generate_ddf_from_label(data_path_calib,device)
    
    # for loop all the scans
    GPE,GLE,LPE,LLE,time_elapsed,i_scan=[],[],[],[],[],1
    for scan_name in list(dataset_keys.keys()):
        frames = h5py.File(os.path.join(data_path_frame,scan_name.split('__')[0][3:],scan_name.split('__')[1]+'.h5'), 'r')['frames'][()]
        landmark = h5py.File(os.path.join(data_path_landmark,"landmark_%03d.h5" %int(scan_name.split('__')[0][3:])), 'r')[scan_name.split('__')[1]][()]
        
        # generate predicted DDF 
        start = time.time()
        pred_GP,pred_GL,pred_LP,pred_LL = predict_ddfs(frames,landmark,data_path_calib,device)
        end = time.time()
        time_elapsed.append((end - start))
        
        # ground truth DDF
        tforms = h5py.File(os.path.join(data_path_transf,scan_name.split('__')[0][3:],scan_name.split('__')[1]+'.h5'), 'r')['tforms'][()]
        labels_GP,labels_GL,labels_LP,labels_LL = generate_GT_ddf.calculate_GT_DDF(frames,tforms,landmark) 

        # plot scan     
        # plot_scans(frames,scan_name,labels_GP,pred_GP,data_path_plot,generate_GT_ddf.tform_calib_scale.cpu(),generate_GT_ddf.image_points.cpu())
        
        # calculate metric
        GPE.append(cal_dist(labels_GP,pred_GP,'all'))
        GLE.append(cal_dist(labels_GL,pred_GL,'landmark'))
        LPE.append(cal_dist(labels_LP,pred_LP,'all'))
        LLE.append(cal_dist(labels_LL,pred_LL,'landmark'))

        print('%4dth scan: %s is finished.'%(i_scan,scan_name))
        i_scan+=1

    # save results into .h5 file
    GPE,GLE,LPE,LLE,time_elapsed = np.array(GPE),np.array(GLE),np.array(LPE),np.array(LLE),np.array(time_elapsed)
    metrics = h5py.File(os.path.join(data_path_metric,"metrics.h5"),'a')
    metrics.create_dataset('GPE', len(GPE), dtype=GPE.dtype, data=GPE)
    metrics.create_dataset('GLE', len(GLE), dtype=GLE.dtype, data=GLE)
    metrics.create_dataset('LPE', len(LPE), dtype=LPE.dtype, data=LPE)
    metrics.create_dataset('LLE', len(LLE), dtype=LLE.dtype, data=LLE)
    metrics.create_dataset('time_elapsed', len(time_elapsed), dtype=time_elapsed.dtype, data=time_elapsed)
    metrics.flush()
    metrics.close()

if __name__ == "__main__":
    main()