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


def get_transforms_image_mm(tforms, lengths, tform_image_mm_to_tool):
    """
    Compute frame-to-frame transformation in image (mm) space.

    Args:
        tforms: (B, N, 4, 4), per-frame tool-to-world transforms
        lengths: (B,), each sample's valid frame count
        tforms_image_mm_to_tool: (4, 4), fixed conjugate transform

    Returns:
        tforms_image1_to_image0_mm: (B, N-1, 4, 4), only up to lengths[i]-1 are valid per sample
    """
    B, N, _, _ = tforms.shape
    tforms_inv = torch.linalg.inv(tforms)  # (B, N, 4, 4)

    # Prepare storage for output
    tforms_image1_to_image0_mm = torch.zeros(B, N - 1, 4, 4, device=tforms.device, dtype=tforms.dtype)

    # Inverse of image→tool matrix (shared for all)
    tform_tool_to_image_mm = torch.linalg.inv(tform_image_mm_to_tool)  # (4, 4)

    for i in range(B):
        n = lengths[i]
        # Compute valid (n-1) pairs
        tform_tool1_to_tool0 = torch.matmul(
            tforms_inv[i, :n-1],    # world→tool0
            tforms[i, 1:n]          # tool1→world
        )  # shape: (n-1, 4, 4)

        # Convert from tool space to image space using conjugation:
        # image1→image0 = T_tool→image @ T_tool1→tool0 @ T_image→tool
        image1_to_image0_mm = tform_tool_to_image_mm @ tform_tool1_to_tool0 @ tform_image_mm_to_tool

        tforms_image1_to_image0_mm[i, :n-1] = image1_to_image0_mm

    return tforms_image1_to_image0_mm


def params_to_transforms(params: torch.Tensor, lengths: torch.Tensor, fill_identity: bool = True) -> torch.Tensor:
    """
    Convert 6D params (Euler angles + translation) into 4x4 homogeneous matrices.
    Args:
        params: (B, max_n-1, 6) predicted params
        lengths: (B,) actual number of valid pairs (n - 1) per sample
        fill_identity: if True, fill padded slots with identity; else zero
    Returns:
        matrix4x4: (B, max_n-1, 4, 4), per-frame transforms
    """
    B, max_n_minus_1, _ = params.shape
    device = params.device
    dtype = params.dtype

    # Preallocate output (default to identity or zeros)
    if fill_identity:
        matrix4x4 = torch.eye(4, dtype=dtype, device=device).view(1, 1, 4, 4).repeat(B, max_n_minus_1, 1, 1)
    else:
        matrix4x4 = torch.zeros(B, max_n_minus_1, 4, 4, dtype=dtype, device=device)

    for i in range(B):
        n = lengths[i].item()
        if n == 0:
            continue  # skip invalid samples

        param_i = params[i, :n]  # (n, 6)
        R = tf.euler_angles_to_matrix(param_i[..., :3], convention='ZYX')  # (n, 3, 3)
        t = param_i[..., 3:].unsqueeze(-1)                                # (n, 3, 1)
        Rt = torch.cat([R, t], dim=2)                                     # (n, 3, 4)
        bottom_row = torch.tensor([0, 0, 0, 1], dtype=dtype, device=device).view(1, 1, 4).expand(n, 1, 4)  # (n, 1, 4)
        matrix4x4[i, :n] = torch.cat([Rt, bottom_row], dim=1)             # (n, 4, 4)

    return matrix4x4


def accumulate_transforms_image_mm(tforms_image1_to_image0_mm, tforms_image2_to_image1_mm): 
    tforms_image2_to_image0_mm = torch.matmul(tforms_image1_to_image0_mm, tforms_image2_to_image1_mm)
    return tforms_image2_to_image0_mm

def get_network_pred_transforms(frames, network, data_pairs, num_samples):
    """
    Predict relative (local) and cumulative (global) transforms from a sequence of frames using a neural network.
    
    Assumes only the first pair in `data_pairs` is used repeatedly (i.e., fixed interval).
    
    Args:
        frames (torch.Tensor): shape=(1, N, H, W), input image sequence with batch size = 1
        network (nn.Module): model that takes a sequence of frames and outputs transform parameters
        data_pairs (torch.Tensor): shape=(M, 2), first row [f0, f1] determines the sampling interval
        num_samples (int): number of frames passed to the network at each step

    Returns:
        tforms_global (torch.Tensor): shape=(N-1, 4, 4), transform from each frame to the first frame
        tforms_local (torch.Tensor): shape=(N-1, 4, 4), transform from current frame to previous frame
    """
    _, num_frames, _, _ = frames.shape
    device = frames.device

    tforms_global = torch.zeros((num_frames - 1, 4, 4), device=device)
    tforms_local = torch.zeros((num_frames - 1, 4, 4), device=device)

    # Initial global transform is identity
    cumulative_transform = torch.eye(4, device=device)

    # Use only the first pair to determine sampling interval
    interval = data_pairs[0, 1] - data_pairs[0, 0]
    frame_idx = 0

    while frame_idx + num_samples <= num_frames:
        with torch.no_grad():
            # Extract a window of frames: shape=(1, num_samples, H, W)
            frame_window = frames[:, frame_idx:frame_idx + num_samples, :, :]

            # Predict transform parameters from the network
            params = network(frame_window)

            # Convert to 4x4 transformation matrix
            relative_transform = params_to_transforms(params, num_samples - 1)[0, 0, :, :]  # (4, 4)

            # Store local transform and update global transform
            tforms_local[frame_idx] = relative_transform
            cumulative_transform = accumulate_transforms_image_mm(cumulative_transform, relative_transform)
            tforms_global[frame_idx] = cumulative_transform

        frame_idx += interval

    # Fill the remaining frames with identity or last global transform
    if frame_idx < num_frames - 1:
        num_remaining = num_frames - 1 - frame_idx
        tforms_local[frame_idx:] = torch.eye(4, device=device).expand(num_remaining, 4, 4)
        tforms_global[frame_idx:] = cumulative_transform.expand(num_remaining, 4, 4)

    return tforms_global, tforms_local
