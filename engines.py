import torch
from tqdm import tqdm
from tools.transforms import get_transforms_image_mm, params_to_transforms, transforms_to_points

def train_one_epoch(args, dataloader, network, optimizer, 
                    criterion_loss_mse, criterion_loss_pcc, 
                    criterion_metric, ref_points, 
                    tform_image_pixel_to_mm, tform_image_mm_to_tool, 
                    device): 
    network.train() 
    this_epoch_loss, this_epoch_dist = 0, 0

    loop = tqdm(dataloader, desc='Training', leave=False)
    for step, (frames, tforms, lengths) in enumerate(loop):
        frames, tforms, lengths = frames.to(device), tforms.to(device), lengths.to(device) 

        labels = get_transforms_image_mm(tforms, tform_image_mm_to_tool, lengths=lengths) 

        optimizer.zero_grad()
        outputs = network(frames, lengths)
        preds = params_to_transforms(outputs, lengths) 
        
        loss_mse = criterion_loss_mse(preds, labels, lengths) 
        loss_pcc = criterion_loss_pcc(preds, labels, lengths) 
        loss = loss_mse + loss_pcc * 0.5   
        
        loss.backward()
        optimizer.step()

        preds_pts = transforms_to_points(preds.data, ref_points, tform_image_pixel_to_mm) 
        labels_pts = transforms_to_points(labels, ref_points, tform_image_pixel_to_mm)  
        dist = criterion_metric(preds_pts, labels_pts).detach()
    
        this_epoch_loss += loss.item()
        this_epoch_dist += dist

        loop.set_postfix(loss=this_epoch_loss/(step+1), dist=this_epoch_dist/(step+1))

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

    loop = tqdm(dataloader, desc='Validating', leave=False)
    with torch.no_grad():
        for step, (frames, tforms, lengths) in enumerate(loop):
            frames = frames.to(device)
            tforms = tforms.to(device)
            lengths = lengths.to(device)

            labels = get_transforms_image_mm(tforms, tform_image_mm_to_tool, lengths=lengths)
            outputs = network(frames, lengths)
            preds = params_to_transforms(outputs, lengths)

            loss_mse = criterion_loss_mse(preds, labels, lengths)
            loss_pcc = criterion_loss_pcc(preds, labels, lengths)
            loss = loss_mse + 0.5 * loss_pcc

            preds_pts = transforms_to_points(preds, ref_points, tform_image_pixel_to_mm)
            labels_pts = transforms_to_points(labels, ref_points, tform_image_pixel_to_mm)
            dist = criterion_metric(preds_pts, labels_pts)

            val_loss_total += loss.item()
            val_dist_total += dist

            loop.set_postfix(loss=val_loss_total/(step+1), dist=val_dist_total/(step+1))

    val_loss_total /= (step + 1)
    val_dist_total /= (step + 1)

    return val_loss_total, val_dist_total
