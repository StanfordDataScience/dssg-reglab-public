# Dataset class for loading the ground truth files
from torch.utils.data import Dataset
import torch
from torch.nn.functional import one_hot
from PIL import Image
import torchvision
import numpy as np
import glob
from random import random

class drivenDataset(Dataset):
    """ 
    Dataset used to load the driven dataset file images and their masks
    """
    
    def __init__(self, image_path, mask_path, transform=True, p=.5):
        """        
        Parameters
        ----------
        image_path : string
            A filepath of where all the image files are
        mask_path : int
            A filepath of where all the mask files are
        transform : function, optional
            A function that applies transforms to the images. The default is None.
        p : float, optional
            The fraction of the time to do transformation on when loading images
        """
        self.images = glob.glob(image_path + '/*.tif')
        self.masks = glob.glob(mask_path + '/*.tif')        
        self.transform = transform
        self.color_jitter = torchvision.transforms.ColorJitter()
        self.train = False
        self.p = .5
        
    def __len__(self):
        return len(self.images)
    
    def transform_fcn(self, image, mask):
        """        
        Parameters
        ----------
        image : Tensor of the image            
        mask : Tensor of the mask           

        Returns
        -------
        Transformed versions of the image and mask

        """        
        # Horizontal flips
        if random() > self.p:
            image = image.flip(-1)
            mask = mask.flip(-1)
        
        # Vertical flips
        if random() > self.p:
            image = image.flip(-2)
            mask = mask.flip(-2)
            
        # Gaussian Noise
        if random() > self.p:
            image = image + torch.normal(0, .3, size=image.size())
            
        # Color Jitter
        if random() > self.p:
            image = self.color_jitter(image)
        
        return image, mask
        
    
    def __getitem__(self, idx):        
        image = torchvision.transforms.ToTensor()(Image.open(self.images[idx]))
        mask_array = np.array(Image.open(self.masks[idx]))        
        mask = one_hot(torch.from_numpy(mask_array).long(), num_classes=2).permute(2, 0, 1).float()
                
        if self.transform & self.train:            
            image, mask = self.transform_fcn(image, mask)            
                            
        return [image, mask]
    
    def setTrain(self, train):
        """        
        Parameters
        ----------
        train : Bool
            Whether or not the Dataset is the training set in order to
            turn data augmentation on.

        Returns
        -------
        None.

        """
        self.train = train