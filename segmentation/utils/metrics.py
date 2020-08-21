import torch
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, unary_from_softmax

def returnInterUnion(labels, prediction, idx=1):
    """
    Parameters
    ----------
    labels : Tensor
        Tensor of labels with class values.
    prediction : Tensor
        Tensor of the prediction with class values.
    idx : Int
        Index of the desired class

    Returns
    -------
    Returns the intersection and union of two arrays for class. Used to calculate 
    Mean IoU:
        
    The Mean Intersection over Union:
        (Labels & Prediction) / (Labels | Prediction)

    NOTE: Not sure if this is correctly implemented

    """
    # First convert to numpy arrays    
    labelArray = ((labels.flatten() == idx).nonzero()).flatten().cpu().numpy()
    predictionArray = ((prediction.flatten() == idx).nonzero()).flatten().cpu().numpy()
    
    intersection = np.intersect1d(labelArray, predictionArray)
    union = np.union1d(labelArray, predictionArray)  
    #mean_iou = np.sum(intersection) / np.sum(union)
    
    return len(intersection), len(union)

def returnPreReF(conf_matrix, idx):
    """
    Parameters
    ----------
    conf_matrix : Numpy Array
        A numpy array of the confusion matrix
    idx : Int
        A number indicating the class with which you wnat the statistics from

    Returns
    -------
    The precision, recall, and F-score of a given class

    """
    precision = conf_matrix[idx, idx]/ np.sum(conf_matrix[:, idx])
    recall = conf_matrix[idx, idx]/ np.sum(conf_matrix[idx, :])
    if (precision == 0) & (recall == 0):
        f_score = 0
    else:
        f_score = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f_score

def returnCRFmask(img_tensor, prediction, num_classes):
    """
    img_tensor : Tensor (Channel x Width x Length)
        Tensor of the original image that is gpu attached
    prediction : Tensor (Width x Length)
        Softmax output of the model
    num_classes : Int
        Number of prediction classes
    
    Returns
    -------
    A post-processed prediction mask for the image tensor
    
    """        
    # Changes input image into 255 format
    changedInput = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8') 
    
    # Get unary energy of the prediction image
    feat_first = prediction.reshape((num_classes, - 1)).cpu().numpy()
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)        
    d = dcrf.DenseCRF2D(img_tensor.shape[2], img_tensor.shape[1], num_classes) # Create CRF filter
    d.setUnaryEnergy(unary)
    
    # Add original image to CRF
    d.addPairwiseGaussian(sxy=(3, 3), compat=5, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(5, 5), srgb=(2, 2, 2), rgbim=np.ascontiguousarray(changedInput),
                       compat=10,
                       kernel=dcrf.DIAG_KERNEL,
                       normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((img_tensor.shape[1], img_tensor.shape[2])) # Get the new mask    
    res = torch.nn.functional.one_hot(torch.Tensor(res).to(torch.int64), num_classes=num_classes).permute(2, 0, 1) # Make it one hot    
    
    return res