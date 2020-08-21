# Dataset class for loading the ground truth files
from torch.utils.data import Dataset
import pickle
import torch
from torchvision.transforms import ToTensor, ColorJitter
import torch.nn.functional as F
import glob
from random import random

# Pretty close to making things gray-ish earth toneish
SEPIA_FILTER = torch.Tensor([[.393, .769, .189],
                             [.349, .686, .168],
                             [.272, .534, .131]]).transpose(0, 1)

# Roughly size of images from Planet
DOWNSAMPLE_SIZE = 330

class groundTruthDataset(Dataset):
    """ 
    Dataset used to load the ground truth file images and their masks
    """
    
    def __init__(self, filepath, transform=False, p=.5, make_small=False, 
                 ignore_lagoon=False):
        """        
        Parameters
        ----------
        filepath : string
            A filepath of where all the pickle files are
        transform : function, optional
            A function that applies transforms to the images. The default is None.
        p : float from 0 to 1
            Percentage of the time you want to perform data augmentation
        make_small : bool
            Whether or not to downsample images 
        ignore_lagoon: bool
            whether or not to consider the lagoons in cafos

        """
        file_list = glob.glob(filepath + '/*.p')
        image_list = []
        mask_list = []
        for file in file_list:
            with open(file, 'rb') as f:
                # Load the files
                loaded = pickle.load(f)
                image = loaded['image']
                lagoon = torch.Tensor(loaded['masks']['Lagoon'])
                cafo = torch.Tensor(loaded['masks']['CAFO Shed'])
                bg = torch.Tensor(loaded['masks']['BACKGROUND'])
                
                # Downsample if needed
                if make_small:                    
                    image = image.resize((DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE))
                    lagoon = downSampleTensor(lagoon, DOWNSAMPLE_SIZE)
                    cafo = downSampleTensor(cafo, DOWNSAMPLE_SIZE)
                    bg = downSampleTensor(bg, DOWNSAMPLE_SIZE)
                
                image_list.append(ToTensor()(image))                
                
                if ignore_lagoon:
                    # Class order 0 : BACKGROUND, 1 : CAFO Shed
                    mask_list.append(torch.stack([bg, cafo]))
                else:
                    # Class order 0 : BACKGROUND, 1 : CAFO Shed, 2 : Lagoon
                    mask_list.append(torch.stack([bg, cafo, lagoon]))
                                
        self.images = image_list
        self.masks = mask_list
        self.transform = transform
        self.color_jitter = ColorJitter()
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
        
        #if random() > self.p:
        #    image = image + torch.normal(0, .3, size=image.size())
            
        # Color Jitter
        if random() > self.p:
            image = self.color_jitter(image)
            
        # Sepia Filter
        if random() > self.p:
            image = image.transpose(2, 1).permute(2, 1, 0).matmul(SEPIA_FILTER).permute(2, 0, 1)
        
        
        return image, mask
        
    
    def __getitem__(self, idx):
        
        image = self.images[idx]
        mask = self.masks[idx]
        
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
        
def downSampleTensor(array, dsize):
    """
    Parameters
    ----------
    array : Numpy Array
        A numpy array of one of the masks
        
    Returns    
    -------
    Downsampled PyTorch tensor equivalent
    """    
    return F.interpolate(array.unsqueeze(0).unsqueeze(0), 
                  size=(dsize, dsize)).reshape(dsize, dsize)