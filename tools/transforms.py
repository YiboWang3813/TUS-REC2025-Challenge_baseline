import torch 
import pytorch3d.transforms as tf 


def get_reference_image_points(image_size, density=2):
    """
    Generate a grid of 2D reference points in image pixel space.

    Args:
        image_size (tuple): (H, W) height and width of the image.
        density (int or tuple): number of points along (H, W) axes.

    Returns:
        image_points (Tensor): shape (4, N), where each column is a point [x, y, 0, 1]^T in homogeneous coordinates.
                               Points are uniformly spaced in pixel space.
    """
    if isinstance(density, int):
        density = (density, density)

    H, W = image_size
    dy, dx = density

    # Generate evenly spaced pixel coordinates along height and width
    y_coords = torch.linspace(1, H, dy)  # (dy,)
    x_coords = torch.linspace(1, W, dx)  # (dx,)

    # Create all (x, y) coordinate pairs via cartesian product
    yx = torch.cartesian_prod(y_coords, x_coords)  # shape: (dy * dx, 2)
    xy = yx[:, [1, 0]].T  # shape: (2, N), [x, y]^T per column

    N = xy.shape[1]

    # Add z=0 and w=1 to get homogeneous 3D coordinates: (4, N)
    z_row = torch.zeros(1, N, dtype=xy.dtype, device=xy.device)
    w_row = torch.ones(1, N, dtype=xy.dtype, device=xy.device)
    image_points = torch.cat([xy, z_row, w_row], dim=0)

    return image_points  # shape: (4, N)


def transforms_to_points(tforms, image_points, tform_image_pixel_to_mm):
    """
    Apply image-to-mm conversion and sequential frame transformations to reference image points.

    Args:
        tforms (Tensor): shape (B, N-1, 4, 4), transformation matrices between frames.
        image_points (Tensor): shape (4, num_points), pixel-space reference points in homogeneous coordinates.
        tform_image_pixel_to_mm (Tensor): shape (4, 4), matrix to convert pixel to physical mm space.

    Returns:
        points_3d_mm (Tensor): shape (B, N-1, 3, num_points), transformed 3D point positions (in mm).
    """
    # First map image_points from pixel space → mm space: shape (4, N)
    points_mm = torch.matmul(tform_image_pixel_to_mm, image_points)  # (4, N)

    # Then apply transformation matrices: (B, N-1, 4, 4) x (4, N) → (B, N-1, 4, N)
    points_transformed = torch.matmul(tforms, points_mm)  # (B, N-1, 4, N)

    # Discard the homogeneous w=1 row → only keep x, y, z (3, N)
    return points_transformed[:, :, :3, :]  # shape: (B, N-1, 3, N)


def get_transforms_image_mm(tforms, tform_image_mm_to_tool, data_pairs):
    """
    Compute transforms from frame j to frame i in image-mm space, using tool→world poses.

    Args:
        tforms (Tensor): shape (B, N, 4, 4), Ground-truth tool poses for each frame (tool→world).
        tform_image_mm_to_tool (Tensor): shape (4, 4), Calibration matrix from image-mm to tool.
        data_pairs (Tensor): shape (M, 2), Index pairs [i, j] indicating transforms from frame j → i (same for all B).

    Returns:
        tform_imagej_to_imagei (Tensor): shape (B, M, 4, 4), Transform matrices from image_j → image_i in image-mm space.
    """
    B = tforms.shape[0]
    M = data_pairs.shape[0]
    i_idx, j_idx = data_pairs[:, 0], data_pairs[:, 1]  # (M,)

    # Precompute inverse of the calibration matrix
    tform_tool_to_image_mm = torch.linalg.inv(tform_image_mm_to_tool)  # (4, 4)

    # Invert tool poses: world → tool
    tforms_inv = torch.linalg.inv(tforms)  # (B, N, 4, 4)

    # Select i-th and j-th frames from each sample in batch
    tform_i_inv = tforms_inv[:, i_idx]     # (B, M, 4, 4): world → tool_i
    tform_j     = tforms[:, j_idx]         # (B, M, 4, 4): tool_j → world

    # Compute tool_j → tool_i
    tform_toolj_to_tooli = torch.matmul(tform_i_inv, tform_j)  # (B, M, 4, 4)

    # Convert to image-mm space: image_j → image_i
    tform_imagej_to_imagei = (
        tform_tool_to_image_mm @ tform_toolj_to_tooli @ tform_image_mm_to_tool
    )  # (B, M, 4, 4)

    return tform_imagej_to_imagei


def params_to_transforms(params, fill_identity=True):
    """
    Convert 6D motion parameters (Euler angles + translation) to 4x4 transformation matrices.

    Args:
        params (Tensor): shape (B, N-1, 6), Motion parameters for frame pairs. Last dim is [rx, ry, rz, tx, ty, tz].
        fill_identity (bool): If True, initialize all outputs with identity matrices (used for debugging/safety).

    Returns:
        tform_matrix (Tensor): shape (B, N-1, 4, 4), Homogeneous transformation matrices for each frame pair.
    """
    B, N_minus_1, _ = params.shape
    device = params.device
    dtype = params.dtype

    # Initialize result tensor
    if fill_identity:
        tform_matrix = torch.eye(4, dtype=dtype, device=device).view(1, 1, 4, 4).repeat(B, N_minus_1, 1, 1)
    else:
        tform_matrix = torch.zeros(B, N_minus_1, 4, 4, dtype=dtype, device=device)

    # Convert Euler angles to rotation matrices
    R = tf.euler_angles_to_matrix(params[..., :3], convention='ZYX')     # (B, N-1, 3, 3)
    t = params[..., 3:].unsqueeze(-1)                                     # (B, N-1, 3, 1)
    Rt = torch.cat([R, t], dim=-1)                                        # (B, N-1, 3, 4)

    # Append homogeneous bottom row [0, 0, 0, 1]
    bottom = torch.tensor([0, 0, 0, 1], dtype=dtype, device=device).view(1, 1, 1, 4)
    bottom = bottom.expand(B, N_minus_1, 1, 4)                            # (B, N-1, 1, 4)

    tform_matrix = torch.cat([Rt, bottom], dim=2)                         # (B, N-1, 4, 4)
    return tform_matrix


def get_network_pred_transforms(frames, network, num_samples, infer_batch_size=16):
    """
    Predict local and global transforms from a (1, N, H, W) ultrasound sequence using batched sliding windows.

    Args:
        frames (Tensor): shape (1, N, H, W), Input image sequence (batch size must be 1).
        network (nn.Module): Frame-pair regression network that maps (B, N, H, W) → (B, N-1, 6).
        num_samples (int): Number of frames per input chunk (used to form N-1 pairs per chunk).
        infer_batch_size (int): Number of chunks processed in one forward pass.

    Returns:
        tforms_global (Tensor): shape (N-1, 4, 4), Cumulative transforms (frame t → frame 0).
        tforms_local (Tensor): shape (N-1, 4, 4), Local transforms (frame t → frame t-1).
    """
    B, N, H, W = frames.shape
    assert B == 1, "Only supports input batch size 1."
    device = frames.device

    # Extract chunks: [(1, num_samples, H, W), ...]
    chunk_list = []
    for start in range(0, N - num_samples + 1, num_samples):
        chunk = frames[:, start:start + num_samples]  # (1, num_samples, H, W)
        chunk_list.append(chunk)

    # Allocate result containers
    tforms_local = torch.zeros((N - 1, 4, 4), device=device)
    tforms_global = torch.zeros((N - 1, 4, 4), device=device)
    cumulative = torch.eye(4, device=device)
    idx = 0  # current output position

    # Run inference chunk-by-chunk
    with torch.no_grad():
        for batch_start in range(0, len(chunk_list), infer_batch_size):
            # (B', num_samples, H, W)
            batch_chunks = torch.cat(chunk_list[batch_start:batch_start + infer_batch_size], dim=0)
            pred_params = network(batch_chunks)                          # (B', num_samples-1, 6)
            pred_tforms = params_to_transforms(pred_params)             # (B', num_samples-1, 4, 4)

            B_prime, num_pairs, _, _ = pred_tforms.shape
            for b in range(B_prime):
                for j in range(num_pairs):
                    if idx >= N - 1:
                        break
                    tform = pred_tforms[b, j]
                    tforms_local[idx] = tform
                    cumulative = cumulative @ tform
                    tforms_global[idx] = cumulative
                    idx += 1

    # Fill remaining (if total frame count not divisible by chunk size)
    if idx < N - 1:
        num_pad = N - 1 - idx
        tforms_local[idx:] = torch.eye(4, device=device)
        tforms_global[idx:] = cumulative.expand(num_pad, 4, 4)

    return tforms_global, tforms_local
