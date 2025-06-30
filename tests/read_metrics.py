import os 
import h5py


file_path = '/raid/liujie/code_recon/code/TUS-REC2025-Challenge_baseline/results/seq_len2__efficientnet_b1__lr0.0001__pred_type_parameter__label_type_point/metrics.h5'
metrics = h5py.File(file_path, 'r') 

GPE = metrics['GPE'][:]
GLE = metrics['GLE'][:]
LPE = metrics['LPE'][:]
LLE = metrics['LLE'][:]

print(f'GPE: {GPE} \nGLE: {GLE} \nLPE: {LPE} \nLLE: {LLE}')