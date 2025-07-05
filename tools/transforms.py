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

def params_to_transforms(params):
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