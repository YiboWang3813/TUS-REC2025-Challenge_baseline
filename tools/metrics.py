import torch 
from tools.transforms import get_transforms_image_mm, get_network_pred_transforms

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


def get_ddfs_from_gt(tforms, tform_image_mm_to_tool, tform_calib_scale, image_points, landmarks):
    """
    Compute DDFs from ground truth tool poses (both global and local).

    Args:
        tforms (torch.Tensor): shape=(B, N, 4, 4), ground truth tool poses in world space
        tform_image_mm_to_tool (torch.Tensor): shape=(4, 4), fixed transform from image mm space to tool space
        tform_calib_scale (torch.Tensor): shape=(4, 4), pixel-to-mm scaling transform
        image_points (torch.Tensor): shape=(4, H*W), homogeneous pixel coordinates
        landmarks (torch.Tensor): shape=(L, 3), landmark coordinates: (frame_id in 1~N-1, x, y)

    Returns:
        ddf_all_global (np.ndarray): shape=(N-1, 3, H*W), global DDF for all pixels
        ddf_landmarks_global (np.ndarray): shape=(3, L), global DDF for landmarks
        ddf_all_local (np.ndarray): shape=(N-1, 3, H*W), local DDF for all pixels
        ddf_landmarks_local (np.ndarray): shape=(3, L), local DDF for landmarks
    """
    B, N = tforms.shape[:2]
    device = tforms.device

    # Use global data pairs to extract tforms 
    data_pairs_global = get_data_pairs_global(N).to(device)  # shape: (N-1, 2)
    tforms_global = get_transforms_image_mm(tforms, tform_image_mm_to_tool, data_pairs_global).squeeze(0)  # (N-1, 4, 4)

    # Use local data pairs to extract tforms 
    data_pairs_local = get_data_pairs_local(N).to(device)  # (N-1, 2)
    tforms_local = get_transforms_image_mm(tforms, tform_image_mm_to_tool, data_pairs_local).squeeze(0)  # (N-1, 4, 4)

    # DDFs
    ddf_all_global, ddf_landmarks_global = get_ddfs_for_all_and_landmarks(
        tforms_global, tform_calib_scale, image_points, landmarks
    )
    ddf_all_local, ddf_landmarks_local = get_ddfs_for_all_and_landmarks(
        tforms_local, tform_calib_scale, image_points, landmarks
    )

    return ddf_all_global, ddf_landmarks_global, ddf_all_local, ddf_landmarks_local


def get_ddfs_from_network_pred(frames, network, num_samples, infer_batch_size,
                               tform_calib_scale, image_points, landmarks):
    """
    Compute both global and local DDFs (for all pixels and landmarks) using predicted transforms from a neural network.

    Args:
        frames (torch.Tensor): shape=(1, N, H, W), input image sequence (batch size must be 1)
        network (nn.Module): model that outputs relative transform parameters
        num_samples (int): number of frames per inference chunk
        infer_batch_size (int): number of chunks to process per forward pass
        tform_calib_scale (torch.Tensor): (4, 4), pixel-to-mm transform
        image_points (torch.Tensor): (4, H*W), homogeneous image grid
        landmarks (torch.Tensor): (L, 3), (frame_id, x, y)

    Returns:
        ddf_all_global (torch.Tensor): (N-1, 3, H*W), dense displacement field (global)
        ddf_landmarks_global (torch.Tensor): (3, L), landmark DDFs (global)
        ddf_all_local (torch.Tensor): (N-1, 3, H*W), dense displacement field (local)
        ddf_landmarks_local (torch.Tensor): (3, L), landmark DDFs (local)
    """
    # Predict global and local transforms
    tforms_global, tforms_local = get_network_pred_transforms(
        frames, network, num_samples, infer_batch_size
    )

    print(tforms_global.shape, tforms_local.shape) 
    print(torch.cuda.max_memory_allocated()) 

    # Compute global DDFs
    ddf_all_global, ddf_landmarks_global = get_ddfs_for_all_and_landmarks(
        tforms_global, tform_calib_scale, image_points, landmarks
    )

    # Compute local DDFs
    ddf_all_local, ddf_landmarks_local = get_ddfs_for_all_and_landmarks(
        tforms_local, tform_calib_scale, image_points, landmarks
    )

    return ddf_all_global, ddf_landmarks_global, ddf_all_local, ddf_landmarks_local


def cal_dist(label, pred, mode='all'):
    """
    Compute the average Euclidean distance between ground truth and predicted displacement vectors.

    Args:
        label (torch.Tensor):
            - If mode='all': shape=(N-1, 3, H*W), ground truth displacement field for all pixels
            - If mode='landmark': shape=(3, L), ground truth displacement vectors for L landmarks
        pred (torch.Tensor):
            - Same shape as `label`, predicted displacement vectors from the network
        mode (str): either 'all' or 'landmark'
            - 'all': compute pixel-wise mean distance over all frames and pixels
            - 'landmark': compute mean distance over all landmarks

    Returns:
        dist (torch.Tensor): scalar, average Euclidean distance between prediction and ground truth
    """

    if mode == 'all':
        # Compute mean over all pixels and frames
        dist = torch.sqrt(((label - pred) ** 2).sum(axis=1)).mean()
    elif mode == 'landmark':
        # Compute mean over all landmarks
        dist = torch.sqrt(((label - pred) ** 2).sum(axis=0)).mean()

    return dist


def normalize_metrics(ddfs_gt, ddfs_pred):
    """
    Normalize displacement error metrics and compute overall and sub-category scores
    according to the "aggregate then rank" strategy (Maier-Hein et al. 2018).

    Args:
        ddfs_gt (tuple of torch.Tensor): 
            - gt_GP: shape=(N-1, 3, H*W), ground truth global pixel DDF
            - gt_GL: shape=(3, L), ground truth global landmark DDF
            - gt_LP: shape=(N-1, 3, H*W), ground truth local pixel DDF
            - gt_LL: shape=(3, L), ground truth local landmark DDF

        ddfs_pred (tuple of torch.Tensor): 
            Same shapes as ddfs_gt, network-predicted DDFs.

    Returns:
        errors (list of float): [GPE, GLE, LPE, LLE], raw reconstruction errors (lower is better)
        scores (list of float): [final_score, global_rec_score, local_rec_score, pixel_rec_score, landmark_rec_score]
            All scores are normalized to [0, 1], higher is better.
    """
    # Unpack ground truth and prediction DDFs
    gt_GP, gt_GL, gt_LP, gt_LL = ddfs_gt
    pred_GP, pred_GL, pred_LP, pred_LL = ddfs_pred

    # Construct "identity transformation" prediction (i.e., no motion)
    identity_pixel_pred = torch.eye(3, device=gt_GP.device).view(1, -1, 1).expand(gt_GP.shape[0], -1, gt_GP.shape[2])
    identity_landmark_pred = torch.eye(3, device=gt_GL.device).view(-1, 1).expand(-1, gt_GL.shape[1])

    # Compute raw displacement errors (lower is better)
    GPE = cal_dist(gt_GP, pred_GP, mode='all')           # Global pixel error
    GLE = cal_dist(gt_GL, pred_GL, mode='landmark')      # Global landmark error
    LPE = cal_dist(gt_LP, pred_LP, mode='all')           # Local pixel error
    LLE = cal_dist(gt_LL, pred_LL, mode='landmark')      # Local landmark error

    # Compute "worst-case" error: using identity transforms as prediction
    worst_GPE = cal_dist(gt_GP, identity_pixel_pred, mode='all')
    worst_GLE = cal_dist(gt_GL, identity_landmark_pred, mode='landmark')
    worst_LPE = cal_dist(gt_LP, identity_pixel_pred, mode='all')
    worst_LLE = cal_dist(gt_LL, identity_landmark_pred, mode='landmark')

    # Compute "best-case" error: GT vs GT → always zero
    best_GPE = cal_dist(gt_GP, gt_GP, mode='all')  # == 0
    best_GLE = cal_dist(gt_GL, gt_GL, mode='landmark')
    best_LPE = cal_dist(gt_LP, gt_LP, mode='all')
    best_LLE = cal_dist(gt_LL, gt_LL, mode='landmark')

    # Normalize to [0, 1] using inverted scale: (x - worst) / (best - worst)
    eps = 1e-6  # for numerical stability

    GPE_norm = (GPE - worst_GPE) / (best_GPE - worst_GPE + eps)
    GLE_norm = (GLE - worst_GLE) / (best_GLE - worst_GLE + eps)
    LPE_norm = (LPE - worst_LPE) / (best_LPE - worst_LPE + eps)
    LLE_norm = (LLE - worst_LLE) / (best_LLE - worst_LLE + eps)

    # Optional: clip to [0, 1] to avoid numeric drift
    GPE_norm = torch.clamp(GPE_norm, 0, 1)
    GLE_norm = torch.clamp(GLE_norm, 0, 1)
    LPE_norm = torch.clamp(LPE_norm, 0, 1)
    LLE_norm = torch.clamp(LLE_norm, 0, 1)

    # Compute overall final score
    final_score = 0.25 * (GPE_norm + GLE_norm + LPE_norm + LLE_norm)

    # Compute category-wise scores (for reference only)
    global_rec_score = 0.5 * (GPE_norm + GLE_norm)
    local_rec_score = 0.5 * (LPE_norm + LLE_norm)
    pixel_rec_score = 0.5 * (GPE_norm + LPE_norm)
    landmark_rec_score = 0.5 * (GLE_norm + LLE_norm)

    errors = [GPE.item(), GLE.item(), LPE.item(), LLE.item()] 
    scores = [final_score.item(), global_rec_score.item(), local_rec_score.item(),
              pixel_rec_score.item(), landmark_rec_score.item()]

    return errors, scores
