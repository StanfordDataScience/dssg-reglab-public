import rasterio
import numpy as np
import torch
import cv2
import sys
sys.path.insert(0, '../utils')
from metrics import returnInterUnion

def returnPredictions(filepath, model):
    """
    Parameters
    ----------
    filepath : String
        String of the filepath of the tiff image
    model : PyTorch Model object
        PyTorch model used for prediction
        
    Returns
    -------
    A mask of predictions of background/cafo for the image
    """
    with rasterio.open(filepath) as src:
        b, g, r, n = src.read()
    rgb = np.stack((r,g,b), axis=0)
    np_image = ((rgb/rgb.max())*255).astype(np.uint8)
    mask = model(torch.Tensor(np_image).unsqueeze(0))
    _, predictions = torch.max(mask, 1)
    #predictions = (predictions == 1).int()
    return predictions

def returnIou(mask1, mask2, idx=1):
    """
    Parameters
    ----------
    mask1 : PyTorch tensor
        A before mask (1 x W X L)
    mask2 : PyTorch tensor
        An after mask (1 x W X L)
    idx : Int
        Which index to find the IoU for
    Returns
    -------
    Returns the IoU of the two masks for the given class
    """       
    array1 = ((mask1.flatten() == idx).nonzero()).flatten().numpy()
    array2 = ((mask2.flatten() == idx).nonzero()).flatten().numpy()
    intersection = np.intersect1d(array1, array2)
    union = np.union1d(array1, array2)            
    
    return len(intersection)/len(union)
    
def returnCafoDiff(mask1, mask2):
    """
    Parameters
    ----------
    mask1 : PyTorch tensor
        A before mask (1 x W X L)
    mask2 : PyTorch tensor
        An after mask (1 x W X L)
        
    Returns
    -------
    Returns the sheer pixel difference between the two masks
    """
    return torch.sum(mask2 == 1).item() - torch.sum(mask1 == 1).item()

def returnClusterDiff(mask1, mask2):
    """
    Parameters
    ----------
    mask1 : PyTorch tensor
        A before mask (1 x W X L)
    mask2 : PyTorch tensor
        An after mask (1 x W X L)
        
    Returns
    -------
    Returns the difference in contours of the two masks.
    """
    mask1 = mask1.squeeze(0).numpy()
    mask2 = mask2.squeeze(0).numpy()
    _, thres1 = cv2.threshold(np.float32(mask1), 0, 255, 0)
    contours1, _ = cv2.findContours(np.uint8(thres1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    _, thres2 = cv2.threshold(np.float32(mask2), 0, 255, 0)
    contours2, _ = cv2.findContours(np.uint8(thres2), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return len(contours2) - len(contours1)

def returnShiftDiff(mask1, mask2):
    """
    Parameters
    ----------
    mask1 : PyTorch tensor
        A before mask (1 x W X L)
    mask2 : PyTorch tensor
        An after mask (1 x W X L)
        
    Returns
    -------
    First computes and up, down, left, right shift of mask2 by 1 to 3 pixels,
    then compute absolute difference with mask1 for each shift. Then find the 
    pixel-wise minimum between the 12 differences and return the sum.
    
    """
    mask1 = (mask1 == 1).int()
    mask2 = (mask2 == 1).int()
    stacked = mask1.repeat((12, 1, 1))
    
    mask_size = list(mask1.shape)
    mask_size.insert(0, 12)
    shifted_mask = torch.zeros(mask_size)

    for i in range(3):    
        shifted_mask[i, 0:-(i+1), :] = mask2[(i+1):, :] #up shifts
        shifted_mask[i+3, (i+1):, :] = mask2[:-(i+1), :] #down shifts
        shifted_mask[i+6, :, 0:-(i+1)] = mask2[:, (i+1):] #left shifts
        shifted_mask[i+9, :, (i+1):] = mask2[:, :-(i+1)] #right shifts

    total_diff = torch.sum(torch.min(torch.abs(stacked - shifted_mask), (0))[0]).item()

    return total_diff