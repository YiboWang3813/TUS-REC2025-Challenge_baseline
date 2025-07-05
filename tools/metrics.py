import torch 
from tools.transforms import get_transforms_image_mm 

def get_data_pairs_global(num_frames): 
    return torch.tensor([[0, n] for n in range(num_frames)])[1:, :]

def get_data_pairs_local(num_frames): 
    return torch.tensor([[n,n+1] for n in range(num_frames-1)])

def get_ddfs_for_all_and_landmarks(tforms, tform_calib_scale, image_points, landmarks, w=640, h=480):
    """
    Compute displacement fields (DDFs) for all image points and specified landmarks.

    Args:
        tforms (torch.Tensor): shape=(num_pairs, 4, 4) transformation matrices (from current frame to reference frame)
        tform_calib_scale (torch.Tensor): shape=(4, 4) scaling matrix from pixel space to mm space
        image_points (torch.Tensor): shape=(4, num_points) homogeneous coordinates in pixel space
        landmarks (torch.Tensor): shape=(L, 3) with each row (frame_id in 1~N-1, x, y)
        w (int): image width
        h (int): image height

    Returns:
        ddf_all (torch.Tensor): shape=(num_pairs, 3, num_points) displacement field for all points
        ddf_landmarks (torch.Tensor): shape=(3, L) displacement field at landmark locations
    """
    num_pairs = tforms.shape[0]
    num_points = image_points.shape[1]
    assert num_points == h * w, f"Expected num_points = {h*w}, got {num_points}"

    # Project image points into mm space (reference frame)
    points_ref = tform_calib_scale @ image_points  # (4, num_points)

    # Apply all transforms to reference points
    points_ref_batched = points_ref.unsqueeze(0).expand(num_pairs, -1, -1)  # (N-1, 4, num_points)
    points_transformed = tforms @ points_ref_batched  # (N-1, 4, num_points)

    # Compute displacement field (DDF)
    ddf_all = points_transformed[:, :3, :] - points_ref[:3, :].unsqueeze(0)  # (N-1, 3, num_points)

    # Extract landmark DDFs
    ddf_reshaped = ddf_all.view(num_pairs, 3, h, w)  # (N-1, 3, H, W)
    frame_ids = landmarks[:, 0].long() - 1  # (L,)
    x = landmarks[:, 1].long() - 1
    y = landmarks[:, 2].long() - 1

    ddf_landmarks = ddf_reshaped[frame_ids, :, y, x].T  # (3, L)

    return ddf_all, ddf_landmarks

def get_ddfs_from_gt(tforms, tforms_image_mm_to_tool, tform_calib_scale, image_points, landmarks):
    """
    Compute both global and local DDFs (for all pixels and landmarks) using ground truth tool poses.

    Args:
        tforms (torch.Tensor): shape=(B, N, 4, 4), ground truth tool poses in world space
        tforms_image_mm_to_tool (torch.Tensor): shape=(4, 4), fixed transform from image mm space to tool space
        tform_calib_scale (torch.Tensor): shape=(4, 4), pixel-to-mm scaling transform
        image_points (torch.Tensor): shape=(4, H*W), homogeneous pixel coordinates
        landmarks (torch.Tensor): shape=(L, 3), landmark coordinates: (frame_id in 1~N-1, x, y)

    Returns:
        ddf_all_global (np.ndarray): shape=(N-1, 3, H*W), global DDF for all pixels
        ddf_landmarks_global (np.ndarray): shape=(3, L), global DDF for landmarks
        ddf_all_local (np.ndarray): shape=(N-1, 3, H*W), local DDF for all pixels
        ddf_landmarks_local (np.ndarray): shape=(3, L), local DDF for landmarks
    """
    num_frames = tforms.shape[1]

    # Global transforms: current → first
    data_pairs_global = get_data_pairs_global(num_frames)
    tforms_global = get_transforms_image_mm(tforms, data_pairs_global, tforms_image_mm_to_tool).squeeze(0)  # (N-1, 4, 4)

    # Local transforms: current → previous
    data_pairs_local = get_data_pairs_local(num_frames)
    tforms_local = get_transforms_image_mm(tforms, data_pairs_local, tforms_image_mm_to_tool).squeeze(0)  # (N-1, 4, 4)

    # Compute DDFs
    ddf_all_global, ddf_landmarks_global = get_ddfs_for_all_and_landmarks(
        tforms_global, tform_calib_scale, image_points, landmarks
    )

    ddf_all_local, ddf_landmarks_local = get_ddfs_for_all_and_landmarks(
        tforms_local, tform_calib_scale, image_points, landmarks
    )

    return ddf_all_global, ddf_landmarks_global, ddf_all_local, ddf_landmarks_local
