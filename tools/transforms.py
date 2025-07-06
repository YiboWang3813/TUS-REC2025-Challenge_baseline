import torch 
import pytorch3d.transforms

def get_reference_image_points(image_size, density=2):
    if isinstance(density, int):
        density = (density, density)

    y_indices = torch.linspace(1, image_size[0], density[0]) # height 
    x_indices = torch.linspace(1, image_size[1], density[1]) # width 
    
    # Create (y, x) pairs and reorder to (x, y)
    yx = torch.cartesian_prod(y_indices, x_indices)  # (N, 2)
    xy = yx[:, [1, 0]]  # (N, 2)
    
    # Transpose to shape (2, N)
    image_points = xy.t()

    num_points = image_points.shape[1]
    z = torch.zeros(1, num_points, dtype=image_points.dtype, device=image_points.device)
    w = torch.ones(1, num_points, dtype=image_points.dtype, device=image_points.device)
    
    # Concatenate to (4, N)
    image_points = torch.cat([image_points, z, w], dim=0)
    
    return image_points  # shape: (4, num_points)

def transforms_to_points(tforms, tforms_image_pixel_to_mm, image_points): 
    return torch.matmul(
        tforms, 
        torch.matmul(tforms_image_pixel_to_mm, image_points)
    )[:, :, :3, :]

def get_transforms_image_mm(tforms, pairs, tforms_image_mm_to_tool): 
    tforms_inv = torch.linalg.inv(tforms) 

    # compute the transformation matrix from the tool1 to the tool0
    tforms_world_to_tool0 = tforms_inv[:, pairs[:, 0], :, :] 
    tforms_tool1_to_world = tforms[:, pairs[:, 1], :, :]
    tforms_tool1_to_tool0 = torch.matmul(tforms_world_to_tool0, tforms_tool1_to_world)

    # convert the transformation matrix from the tool space to the image space (in mm) using the conjugate transformation
    tforms_tool_to_image_mm = torch.linalg.inv(tforms_image_mm_to_tool) 
    tforms_image1_to_image0_mm = torch.matmul(
        tforms_tool_to_image_mm[None, None, :, :],
        torch.matmul(tforms_tool1_to_tool0, tforms_image_mm_to_tool[None, None, :, :]),
    )

    return tforms_image1_to_image0_mm

def params_to_transforms(params, num_pairs):
    if len(list(params.shape)) == 2: 
        params = params.reshape(params.shape[0], num_pairs, -1) 

    # params: (B, N, 6) -> (rx, ry, rz, tx, ty, tz)
    matrix3x3 = pytorch3d.transforms.euler_angles_to_matrix(params[..., :3], 'ZYX')  # (B, N, 3, 3)
    translation = params[..., 3:].unsqueeze(-1)  # (B, N, 3, 1)
    matrix3x4 = torch.cat([matrix3x3, translation], dim=3)  # (B, N, 3, 4)

    batch_size, num_pairs = matrix3x4.shape[:2]
    device = params.device
    last_row = torch.tensor([0, 0, 0, 1], dtype=params.dtype, device=device)
    last_row = last_row.view(1, 1, 1, 4).expand(batch_size, num_pairs, 1, 4)  # (B, N, 1, 4)

    matrix4x4 = torch.cat([matrix3x4, last_row], dim=2)  # (B, N, 4, 4)
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
