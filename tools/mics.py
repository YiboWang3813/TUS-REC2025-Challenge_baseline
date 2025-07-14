import numpy as np
import torch

def read_calibration_matrices(calib_file_path: str):
    """
    Read calibration matrices from a .csv file.

    Args:
        calib_file_path (str): Path to the calibration CSV file.

    Returns:
        tform_image_pixel_to_mm (torch.Tensor): (4, 4) pixel → mm scaling matrix in image space.
        tform_image_mm_to_tool (torch.Tensor): (4, 4) image(mm) → tool calibration matrix.
    """
    tform_calib = np.empty((8, 4), dtype=np.float32)

    with open(calib_file_path, 'r') as f:
        lines = [line.strip().split(',') for line in f.readlines()]
        tform_calib[0:4, :] = np.array(lines[1:5], dtype=np.float32)   # pixel → mm
        tform_calib[4:8, :] = np.array(lines[6:10], dtype=np.float32)  # image(mm) → tool

    tform_image_pixel_to_mm = torch.tensor(tform_calib[0:4, :])      # (4, 4)
    tform_image_mm_to_tool = torch.tensor(tform_calib[4:8, :])       # (4, 4)

    return tform_image_pixel_to_mm, tform_image_mm_to_tool


def get_n_choices_for_epoch(epoch):
    if epoch < 2:
        return [2]
    elif epoch < 4:
        return [2, 4]
    elif epoch < 6:
        return [2, 4, 8]
    elif epoch < 8: 
        return [2, 4, 8, 16]
    elif epoch < 10: 
        return [2, 4, 8, 16, 32]


def get_batch_size_for_n(n, base_batch=8, base_n=16):
    return max(1, base_batch * base_n // n)
