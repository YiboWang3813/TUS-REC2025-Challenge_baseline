import torch 

from tools.transforms import get_transforms_image_mm, params_to_transforms, transforms_to_points 


def train_one_epoch(args, dataloader, network, optimizer, 
                    criterion_loss_mse, criterion_loss_pcc, 
                    criterion_metric, ref_points, 
                    tform_image_pixel_to_mm, tform_image_mm_to_tool, 
                    device): 
    network.train() 
    this_epoch_loss, this_epoch_dist = 0, 0 
    for step, (frames, tforms, lengths) in enumerate(dataloader):
        frames, tforms, lengths = frames.to(device), tforms.to(device), lengths.to(device) 

        # Transform frame-level gt to pair-level label 
        labels = get_transforms_image_mm(tforms, lengths, tform_image_mm_to_tool) 

        optimizer.zero_grad()

        # Forward the network 
        outputs = network(frames)

        # Transforms network's prediction 
        preds = params_to_transforms(outputs, lengths) 
        
        # Compute Loss 
        loss_mse = criterion_loss_mse(preds, labels) 
        loss_pcc = criterion_loss_pcc(preds, labels) 
        loss = loss_mse + loss_pcc * 0.5   
        
        loss.backward()
        optimizer.step()

        # Convert transforms of prediction and label to points for metric computing 
        preds_pts = transforms_to_points(preds.data, ref_points, tform_image_pixel_to_mm) 
        labels_pts = transforms_to_points(labels, ref_points, tform_image_pixel_to_mm)  
        dist = criterion_metric(preds_pts, labels_pts).detach()
    
        this_epoch_loss += loss.item()
        this_epoch_dist += dist
        
    this_epoch_loss /= (step + 1)
    this_epoch_dist /= (step + 1)

    return this_epoch_loss, this_epoch_dist 


def validate_one_epoch(args, dataloader, network,
                       criterion_loss_mse, criterion_loss_pcc,
                       criterion_metric, ref_points,
                       tform_image_pixel_to_mm, tform_image_mm_to_tool,
                       device):
    network.eval()
    val_loss_total, val_dist_total = 0, 0

    with torch.no_grad():
        for step, (frames, tforms, lengths) in enumerate(dataloader):
            frames = frames.to(device)
            tforms = tforms.to(device)
            lengths = lengths.to(device)

            # Get GT pairwise transforms
            labels = get_transforms_image_mm(tforms, lengths, tform_image_mm_to_tool)

            # Network forward
            outputs = network(frames)
            preds = params_to_transforms(outputs, lengths)

            # Compute validation losses
            loss_mse = criterion_loss_mse(preds, labels)
            loss_pcc = criterion_loss_pcc(preds, labels)
            loss = loss_mse + 0.5 * loss_pcc

            # Metric on 3D point difference
            preds_pts = transforms_to_points(preds, ref_points, tform_image_pixel_to_mm)
            labels_pts = transforms_to_points(labels, ref_points, tform_image_pixel_to_mm)
            dist = criterion_metric(preds_pts, labels_pts)

            val_loss_total += loss.item()
            val_dist_total += dist

    val_loss_total /= (step + 1)
    val_dist_total /= (step + 1)

    return val_loss_total, val_dist_total
