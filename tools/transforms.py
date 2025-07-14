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


def get_transforms_image_mm(
    tforms: torch.Tensor,
    tform_image_mm_to_tool: torch.Tensor,
    lengths: torch.Tensor = None,
    data_pairs: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute transforms between frame pairs in image mm space (either local or global depending on `data_pairs`).

    Args:
        tforms: (B, N, 4, 4), ground truth tool poses (tool→world)
        tform_image_mm_to_tool: (4, 4), fixed matrix from image mm → tool
        lengths: (B,), optional, number of valid frames per sample (only needed for local mode)
        data_pairs: (B, M, 2), optional, for global transform: each row [i, j] means j→i

    Returns:
        tforms_imagej_to_imagei_mm: (B, M, 4, 4), image mm transforms j → i
    """
    B, N, _, _ = tforms.shape
    tform_tool_to_image_mm = torch.linalg.inv(tform_image_mm_to_tool)
    tforms_inv = torch.linalg.inv(tforms)

    if data_pairs is not None:
        # Global (arbitrary) mode: use provided index pairs
        M = data_pairs.shape[1]
        result = torch.zeros((B, M, 4, 4), device=tforms.device, dtype=tforms.dtype)

        for b in range(B):
            for m in range(M):
                i, j = data_pairs[b, m]
                T_toolj_to_tooli = torch.matmul(tforms_inv[b, i], tforms[b, j])
                T_imagej_to_imagei = tform_tool_to_image_mm @ T_toolj_to_tooli @ tform_image_mm_to_tool
                result[b, m] = T_imagej_to_imagei

    else:
        # Local mode: adjacent frame pairs
        assert lengths is not None, "lengths must be provided when data_pairs is None"
        result = torch.zeros((B, N - 1, 4, 4), device=tforms.device, dtype=tforms.dtype)

        for b in range(B):
            n = lengths[b]
            T_tool1_to_tool0 = torch.matmul(tforms_inv[b, :n - 1], tforms[b, 1:n])
            T_image1_to_image0 = tform_tool_to_image_mm @ T_tool1_to_tool0 @ tform_image_mm_to_tool
            result[b, :n - 1] = T_image1_to_image0

    return result


def params_to_transforms(params: torch.Tensor, lengths: torch.Tensor, fill_identity: bool = True) -> torch.Tensor:
    """
    Convert 6D params (Euler angles + translation) into 4x4 homogeneous matrices.
    Args:
        params: (B, max_n-1, 6) predicted params
        lengths: (B,) actual number of valid pairs per sample, directly from dataloader and need to minus 1
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

        param_i = params[i, :n-1]  # (n, 6)
        R = tf.euler_angles_to_matrix(param_i[..., :3], convention='ZYX')  # (n-1, 3, 3)
        t = param_i[..., 3:].unsqueeze(-1)                                # (n-1, 3, 1)
        Rt = torch.cat([R, t], dim=2)                                     # (n-1, 3, 4)
        bottom_row = torch.tensor([0, 0, 0, 1], dtype=dtype, device=device).view(1, 1, 4).expand(n-1, 1, 4)  # (n-1, 1, 4)
        # print(Rt.shape, bottom_row.shape)
        matrix4x4[i, :n] = torch.cat([Rt, bottom_row], dim=1)             # (n-1, 4, 4)

    return matrix4x4


def get_network_pred_transforms(frames: torch.Tensor, network: torch.nn.Module, num_samples: int, infer_batch_size: int = 16):
    """
    Predict local and global transforms from an ultrasound image sequence using batched sliding windows.

    Args:
        frames: (1, N, H, W), input sequence
        network: model to predict 6D params from frame chunks
        num_samples: number of frames in each chunk
        infer_batch_size: number of chunks processed in one forward pass

    Returns:
        tforms_global: (N-1, 4, 4)
        tforms_local: (N-1, 4, 4)
    """
    B, N, H, W = frames.shape
    assert B == 1, "Only supports input batch size 1."
    device = frames.device

    # Divide into chunks of size `num_samples`
    chunk_list = []
    valid_starts = []
    for i in range(0, N - num_samples + 1, num_samples):
        chunk = frames[:, i:i + num_samples]  # shape: (1, num_samples, H, W)
        chunk_list.append(chunk)
        valid_starts.append(i)

    # Track results
    tforms_global = torch.zeros((N - 1, 4, 4), device=device)
    tforms_local = torch.zeros((N - 1, 4, 4), device=device)
    cumulative = torch.eye(4, device=device)
    idx = 0  # current position in output

    # Batch inference loop
    with torch.no_grad():
        for batch_start in range(0, len(chunk_list), infer_batch_size):
            batch_chunks = torch.cat(chunk_list[batch_start:batch_start + infer_batch_size], dim=0)  # (B', num_samples, H, W)
            lengths = torch.full((batch_chunks.shape[0],), num_samples - 1, dtype=torch.long, device=device)
            pred_params = network(batch_chunks, lengths)  # (B', num_samples-1, 6)
            pred_tforms = params_to_transforms(pred_params, lengths)  # (B', num_samples-1, 4, 4)

            for b in range(pred_tforms.shape[0]):
                for j in range(num_samples - 1):
                    if idx >= N - 1:
                        break
                    tform = pred_tforms[b, j]
                    tforms_local[idx] = tform
                    cumulative = torch.matmul(cumulative, tform) 
                    tforms_global[idx] = cumulative
                    idx += 1

    # Handle leftover (frames not covered by complete chunk)
    if idx < N - 1:
        num_pad = N - 1 - idx
        tforms_local[idx:] = torch.eye(4, device=device).expand(num_pad, 4, 4)
        tforms_global[idx:] = cumulative.expand(num_pad, 4, 4)

    return tforms_global, tforms_local
