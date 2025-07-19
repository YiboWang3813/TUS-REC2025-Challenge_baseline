import os
import time
import argparse
import h5py
import numpy as np
import torch

from tools.networks import build_network
from tools.transforms import get_reference_image_points
from tools.metrics import (
    get_ddfs_from_gt,
    get_ddfs_from_network_pred,
    normalize_metrics
)
from tools.mics import read_calibration_matrices


def main():
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Set test dataset path
    test_data_dir = '/raid/liujie/code_recon/data/ultrasound/Freehand_US_data_val_2025'
    dataset_keys = h5py.File(os.path.join(test_data_dir, "dataset_keys.h5"), 'r')

    # Set experiment directory and load model
    exp_dir = '/raid/liujie/code_recon/checkpoints/frame_pair_model_fold0'

    parser = argparse.ArgumentParser()
    parser.add_argument('--network_name', type=str, default='frame_pair_model', help='Name of the model architecture')
    parser.add_argument('--image_shape', type=str, default='480,640', help='Image shape as H,W (e.g. 480,640)')
    args = parser.parse_args()

    best_weight_path = os.path.join(exp_dir, 'weights', f'{args.network_name}_best.pth')

    network = build_network(args)
    network.load_state_dict(torch.load(best_weight_path, map_location=device))
    network.to(device)
    print('Network loaded.')

    # Load calibration matrices
    calib_matrix_path = os.path.join(test_data_dir, 'calib_matrix.csv')
    tform_image_pixel_to_mm, tform_image_mm_to_tool = read_calibration_matrices(calib_matrix_path)
    tform_image_pixel_to_mm = tform_image_pixel_to_mm.to(device)
    tform_image_mm_to_tool = tform_image_mm_to_tool.to(device)

    # Build reference points for metric computation
    image_shape = [int(n) for n in args.image_shape.split(',')]
    all_ref_points = get_reference_image_points(image_shape, image_shape)
    all_ref_points = all_ref_points.to(device)

    print('Preparation done.')

    all_errors, all_scores, all_infer_time = [], [], []

    for scan_name in list(dataset_keys.keys()):
        # Parse scan name
        # print(scan_name)
        begin_time = time.time() 
        scan_name = scan_name.split('__')
        subject_id_str, object_type = scan_name[0], scan_name[1]
        subject_id = int(subject_id_str[3:])

        # Load image frames
        t0 = time.time() 
        frame_filename = f'{subject_id:03d}/{object_type}.h5'
        frame_path = os.path.join(test_data_dir, 'frames', frame_filename)
        frames = h5py.File(frame_path, 'r')['frames'][()]
        print(f'load frame time cost: {(time.time() - t0):.3f} s') 

        # Load ground truth transforms
        t1 = time.time() 
        tform_filename = f'{subject_id:03d}/{object_type}.h5'
        tform_path = os.path.join(test_data_dir, 'transfs', tform_filename)
        tforms = h5py.File(tform_path, 'r')['tforms'][()]
        print(f'load tform time cost: {(time.time() - t1):.3f} s') 

        # Load landmarks
        t2 = time.time() 
        landmark_filename = f'landmark_{subject_id:03d}.h5'
        landmark_path = os.path.join(test_data_dir, 'landmarks', landmark_filename)
        landmarks = h5py.File(landmark_path, 'r')[object_type][()]
        print(f'load landmarks time cost: {(time.time() - t2):.3f} s') 

        # Normalize inputs 
        t3 = time.time() 
        frames = torch.from_numpy((frames / 255.0).astype(np.float32)).unsqueeze(0)  # shape: (1, N, H, W)
        tforms = torch.from_numpy(tforms).unsqueeze(0)          # shape: (1, N, 4, 4)
        landmarks = torch.from_numpy(landmarks)                 # shape: (100, 3) 
        print(f'move data to GPU, time cost: {(time.time() - t3):.3f} s')

        # Compute GT DDFs
        start_time = time.time()
        ddfs_gt = get_ddfs_from_gt(
            tforms, tform_image_mm_to_tool, tform_image_pixel_to_mm, all_ref_points, landmarks
        )
        end_time = time.time()
        print(f'Generate DDFs from gt, time cost: {(end_time - start_time):.3f} s') 

        # Compute predicted DDFs
        start_time = time.time()
        ddfs_pred = get_ddfs_from_network_pred(
            frames, network, 2, 32, tform_image_pixel_to_mm, all_ref_points, landmarks 
        )
        end_time = time.time()
        infer_time = end_time - start_time
        print(f'Generate DDFs from network, time cost: {(end_time - start_time):.3f} s')

        # Compute errors and metrics
        errors, scores = normalize_metrics(ddfs_gt, ddfs_pred)

        # Save results
        all_errors.append(errors)
        all_scores.append(scores)
        all_infer_time.append(infer_time)

        print(f'{scan_name} done, errors: {errors}, scores: {scores}')

    # Save final evaluation results
    results = {
        'all_errors': np.array(all_errors),
        'all_scores': np.array(all_scores),
        'all_infer_time': np.array(all_infer_time)
    }

    results_path = os.path.join(exp_dir, 'errors_scores.h5')
    with h5py.File(results_path, 'a') as f:
        for k, v in results.items():
            f.create_dataset(k, data=v)
        f.flush()


if __name__ == "__main__":
    main()
