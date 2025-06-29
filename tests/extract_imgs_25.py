# import os 
# import h5py 

# from PIL import Image 


# image_dir = '/raid/liujie/code_recon/data/ultrasound/Freehand_US_data_train_2025/frames_transfs'
# output_dir = '/raid/liujie/code_recon/results/imgs_train_25'
# scan_names = ['LH_rotation.h5', 'RH_rotation.h5']

# subject_names = os.listdir(image_dir) 
# subject_names = sorted(subject_names) # ['000', '001', ..., '049'] 

# for i in range(len(subject_names)): 
#     if i == 1: 
#         break 
#     subject_dir = os.path.join(image_dir, subject_names[i]) 
#     output_dir_subject = os.path.join(output_dir, subject_names[i]) 
#     os.makedirs(output_dir_subject, exist_ok=True) 

#     for j in range(len(scan_names)): 
#         scan_path = os.path.join(subject_dir, scan_names[j])
#         output_dir_scan = os.path.join(output_dir_subject, scan_names[j][:-3]) 
#         os.makedirs(output_dir_scan, exist_ok=True)
#         h5file = h5py.File(scan_path, 'r')
#         frames = h5file['frames']

#         for n in range(frames.shape[0]): 
#             frame = frames[n, ...] 
#             frame = Image.fromarray(frame) 
#             frame_save_path = os.path.join(output_dir_scan, f'frame_{n:04d}.png')
#             frame.save(frame_save_path)
    
#     print(f'subject {i} done') 


import os
import h5py
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# 路径参数
image_dir = '/raid/liujie/code_recon/data/ultrasound/Freehand_US_data_train_2025/frames_transfs'
output_dir = '/raid/liujie/code_recon/results/imgs_train_25'
scan_names = ['LH_rotation.h5', 'RH_rotation.h5']

# 获取所有 subject
subject_names = sorted(os.listdir(image_dir))  # ['000', '001', ..., '049']

# 线程池 worker 数
NUM_THREADS = 8  # 你可以根据 CPU 核心数和 I/O 压力调整这个值

# 单个扫描处理函数
def process_scan(subject_name, scan_name):
    subject_dir = os.path.join(image_dir, subject_name)
    output_dir_subject = os.path.join(output_dir, subject_name)
    os.makedirs(output_dir_subject, exist_ok=True)

    scan_path = os.path.join(subject_dir, scan_name)
    output_dir_scan = os.path.join(output_dir_subject, scan_name[:-3])
    os.makedirs(output_dir_scan, exist_ok=True)

    try:
        with h5py.File(scan_path, 'r') as h5file:
            frames = h5file['frames']
            for n in range(frames.shape[0]):
                frame = Image.fromarray(frames[n, ...])
                frame_save_path = os.path.join(output_dir_scan, f'frame_{n:04d}.png')
                frame.save(frame_save_path)
        print(f"{subject_name}/{scan_name} done")
    except Exception as e:
        print(f"Failed processing {subject_name}/{scan_name}: {e}")

# 主线程：分配任务给线程池
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = []
    for subject_name in subject_names:
        for scan_name in scan_names:
            futures.append(executor.submit(process_scan, subject_name, scan_name))

    # 等待所有任务完成
    for future in as_completed(futures):
        future.result()  # 如果有异常，会在这里 raise 出来
