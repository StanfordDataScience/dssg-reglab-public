from torch.utils.data import DataLoader, random_split
import torch

def splitDataset(dataset, split_prop=[.7, .15, .15]):
    """
    Parameters
    ----------
    dataset : String
        A Dataset PyTorch object
        
    split_prop : List of decimals
        What proportion of training splits desired. Default is 70-15-15

    Returns
    -------
    A list of the three datasets of train, valid, test

    """
    # For reproducibility    
    torch.manual_seed(0)    
    dataset_size = len(dataset)
    splits = [dataset_size*split_prop[0], dataset_size*split_prop[1], dataset_size*split_prop[2]]
    splits = [int(round(split,0)) for split in splits]
    
    # If sum doesn't add up evenly, just add to test set
    if sum(splits) != dataset_size:
        splits[2] += dataset_size - sum(splits)
    
    train, valid, test = random_split(dataset, splits)
    train.dataset.setTrain(True) #Apply transforms to training set
    
    return [train, valid, test]

def returnLoaders(datasets, batch_size, shuffle):
    """
    Parameters
    ----------
    datasets : List
        List of the three datasets of train, valid, test.
        
    batch_size: Int
        Desired batch size of the dataloaders.
        
    shuffle: Bool
        Whether or not the dataset should be shuffled.

    Returns
    -------
    A list of the three dataloders of train, valid, test.

    """
    train, valid, test = datasets
    trainLoader = DataLoader(train, batch_size, shuffle=shuffle)
	#No need to shuffle valid or test
    validLoader = DataLoader(valid, batch_size, shuffle=False) 
    testLoader = DataLoader(test, batch_size, shuffle=False)
    
    return [trainLoader, validLoader, testLoader]