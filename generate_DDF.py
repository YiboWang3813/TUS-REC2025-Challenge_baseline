
# This script is an example to show how to generate the 4 DDFs.
# The DDFs could further be used in loss functions.

import os
import torch
import h5py
import time
from utils.loader import Dataset
from utils.plot_functions import *
from utils.funs import *
from utils.generate_ddf_from_label import generate_ddf_from_label
from utils.predict_ddfs import predict_ddfs
from utils.plot_scans import plot_scans
from utils.metrics import cal_dist

# load configs
saved_model_path = os.getcwd()+ '/results/seq_len2__efficientnet_b1__lr0.0001__pred_type_parameter__label_type_point'
_,opt = load_config(saved_model_path+'/config.txt')
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# test set loader
opt.FILENAME_TEST=opt.FILENAME_TEST+'.json'
dset_test = Dataset.read_json(os.path.join(os.getcwd(),opt.SAVE_PATH,opt.FILENAME_TEST),num_samples = -1)

# save results
GPE,GLE,LPE,LLE,time_elapsed=[],[],[],[],[]
saved_folder_test = os.path.join(os.getcwd(),opt.SAVE_PATH, 'testing','testing_test_results')
if not os.path.exists(saved_folder_test):
    os.makedirs(saved_folder_test)

generate_GT_ddf = generate_ddf_from_label(os.path.join(os.path.dirname(os.path.realpath(__file__)),opt.FILENAME_CALIB),device)

for scan_index in range(len(dset_test)):

    # load the scan and landmark
    frames, tforms, indices, scan_name = dset_test[scan_index]
    landmark = h5py.File(os.path.join(os.getcwd(),opt.LANDMARK_PATH,"landmark_%03d.h5" %indices[0]), 'r')[scan_name][()]

    # generate four ground truth DDFs
    labels_GP,labels_GL,labels_LP,labels_LL = generate_GT_ddf.calculate_GT_DDF(frames,tforms,landmark) 

    # generate four predicted DDFs
    start = time.time()
    pred_GP,pred_GL,pred_LP,pred_LL = predict_ddfs(frames,landmark,opt.FILENAME_CALIB,device)
    torch.cuda.synchronize()
    end = time.time()
    time_elapsed.append(end - start)

    # plot scan     
    plot_scans(frames,scan_name,indices,labels_GP,pred_GP,saved_folder_test,generate_GT_ddf.tform_calib_scale.cpu(),generate_GT_ddf.image_points.cpu())
    
    # calculate metric
    GPE.append(cal_dist(labels_GP,pred_GP,'all'))
    GLE.append(cal_dist(labels_GL,pred_GL,'landmark'))
    LPE.append(cal_dist(labels_LP,pred_LP,'all'))
    LLE.append(cal_dist(labels_LL,pred_LL,'landmark'))

    print('%4dth scan: %s is finished.'%(scan_index,scan_name))


# save results into .h5 file
GPE,GLE,LPE,LLE,time_elapsed = np.array(GPE),np.array(GLE),np.array(LPE),np.array(LLE),np.array(time_elapsed)
metrics = h5py.File(os.path.join(opt.SAVE_PATH,"metrics.h5"),'a')
metrics.create_dataset('GPE', len(GPE), dtype=GPE.dtype, data=GPE)
metrics.create_dataset('GLE', len(GLE), dtype=GLE.dtype, data=GLE)
metrics.create_dataset('LPE', len(LPE), dtype=LPE.dtype, data=LPE)
metrics.create_dataset('LLE', len(LLE), dtype=LLE.dtype, data=LLE)
metrics.create_dataset('time_elapsed', len(time_elapsed), dtype=time_elapsed.dtype, data=time_elapsed)
metrics.flush()
metrics.close()