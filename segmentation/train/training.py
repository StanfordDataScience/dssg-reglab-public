import torch
from torch.nn import BCEWithLogitsLoss
import numpy as np
from sklearn.metrics import confusion_matrix
import random
from tqdm import tqdm
import argparse
from datetime import date
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(0, '../utils')
from ground_truth_dataset import groundTruthDataset
from driven_dataset import drivenDataset
from data_functions import splitDataset, returnLoaders
from metrics import returnInterUnion, returnPreReF
sys.path.insert(0, '../models')
from unet_model import UNet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', required=False, help="Which dataset to train on", default="driven")
parser.add_argument('--image_path', '-ip', help="if driven dataset, path of images")
parser.add_argument('--mask_path', '-mp', help="if driven dataset, path of mask")
parser.add_argument('--filepath', '-f', required=False, help="Path of the folder with the images.", 
                    default="../../../segmentation_ground_truth")
parser.add_argument('--optimizer', '-opt', help="Which optimizer to use", nargs='?', type=str, default="adam")
parser.add_argument('--model', '-m', help="which model to use", default="unet")
parser.add_argument('--lr', '-lr', help="Learning Rate", nargs='?', type=float, default=3e-3)
parser.add_argument('--wd', '-wd', help="Weight Decay", nargs='?', type=float, default=0)
parser.add_argument('--momentum', '-mo', help="Momentum", nargs='?', type=float, default=9e-1)
parser.add_argument('--step_size', '-step', help="Step size for Learning Rate decay", default=0)
parser.add_argument('--epochs', '-e', help="Number of Epochs", nargs='?', type=int, default=5)
parser.add_argument('--shuffle', '-s', help="Whether to shuffle the dataset", nargs='?', 
                    type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--batch_size', '-bs', help="Batch Size", nargs='?', type=int, default=16)
parser.add_argument('--save_model', '-sm', help="Whether to save the model", nargs='?', 
                    type=lambda x: (str(x).lower() == 'true'), default=True)
args = parser.parse_args()

# Constants
NUM_CLASSES = 3 # For ground truth data, there's 3 classes of Background, Lagoon, CAFO

def train_one_epoch(epoch, train_num_batches, model, device, trainloader, epoch_pbar, 
                    optimizer, writer, criterion):
    """
    Parameters
    ----------
    epoch : int
        The current epoch of training.
    train_num_batches : Int
        Number of batches in trainloader.
    model : Model
        Model used to evaluate.
    device : Torch Object
        Whether or not we're using gpu/cpu.
    trainloader : DataLoader
        Dataloader for the training set.
    epoch_pbar : progress bar object
        Used to print out information to console.
    optimizer : Optimizer object
        Optimizer of training.
    writer : TensorBoard object
        Used to write information to tensorboard.
    criterion : Loss function
        Loss function used to evaluate.    

    Returns
    -------
    Training loss, the number of correctly classified examples, and the
    mean IoU.
    
    NOTE: Calculation of confusion matrix is not done in training for 
    efficiency purposes

    """
    train_loss = []
    train_correct = 0
    train_denom = 0
    acc_loss = 0
    acc_avg = 0
    intersection = 0
    union = 0
    
    for i, batch in enumerate(trainloader):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
    
        #Forward pass            
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc_loss += loss.item()                
        train_loss.append(loss.item())     
                
        #Pixel by Pixel overall accuracy                
        _, predictions = torch.max(outputs, 1) #the value with highest logit is the predicted class
        train_correct += torch.sum(predictions == torch.argmax(labels, 1)).item()
        train_denom += (inputs.shape[-1] * inputs.shape[-2] * inputs.shape[0]) #The number of pixels per each channel
                
        #Calculating IoU
        labels_unhot = torch.argmax(labels, 1)
        curr_int, curr_uni = returnInterUnion(labels_unhot, predictions)
        intersection += np.sum(curr_int)
        union += np.sum(curr_uni)
                                
        #Update progress bar
        avg_loss = acc_loss/(i + 1)                
        acc_avg = train_correct/train_denom
        desc = f"Epoch {epoch} - loss {avg_loss:.4f} - acc {acc_avg:.4f} - Mean IoU {(intersection/union):.4f}" #lr {optimizer.param_groups[0]['lr']}"
        epoch_pbar.set_description(desc)
        epoch_pbar.update(1)
        
        #Compute gradient and update params
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #clip gradient
        optimizer.step()                        
        optimizer.zero_grad()
        model.zero_grad()
        
        #Write to Tensorboard
        writer.add_scalar('Iteration Training Loss', loss.item(), 
                              epoch*len(trainloader) + i + 1)
    
    return train_loss, train_correct, intersection/union

def valid_one_epoch(epoch, valid_num_batches, model, device, validloader, epoch_pbar, 
                    optimizer, writer, criterion, conf_matrix, class_list):
    """
    Parameters
    ----------
    epoch : int
        The current epoch of training.
    train_num_batches : Int
        Number of batches in trainloader.
    model : Model
        Model used to evaluate.
    device : Torch Object
        Whether or not we're using gpu/cpu.
    trainloader : DataLoader
        Dataloader for the training set.
    epoch_pbar : progress bar object
        Used to print out information to console.
    optimizer : Optimizer object
        Optimizer of training.
    writer : TensorBoard object
        Used to write information to tensorboard.
    criterion : Loss function
        Loss function used to evaluate.    
    conf_matrix : Tensor
        A confusion matrix to fill out for more information.
    class_list : List
        A list that outlines what classes there are for the confusion matrix

    Returns
    -------
    Validation loss, the number of correctly classified examples, 
    the mean IoU along with confusion matrix.

    """
    valid_loss = []
    valid_correct = 0
    valid_denom = 0
    acc_loss = 0
    acc_avg = 0
    intersection = 0
    union = 0
    cafo_int = 0
    cafo_union = 0
    
    for i, batch in enumerate(validloader):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        
        #Forward pass
        with torch.no_grad():
            outputs = model(inputs)            
        loss = criterion(outputs, labels)
        acc_loss += loss.item()
        valid_loss.append(loss.item())        
        
        #Calculating accuracy
        _, predictions = torch.max(outputs, 1) #the value with highest logit is the predicted class
        labels_unhot = torch.argmax(labels, 1)
        valid_correct += torch.sum(predictions == labels_unhot).item()
        valid_denom += (inputs.shape[-1] * inputs.shape[-2] * inputs.shape[0]) #The number of pixels per each channel
        
        #Calculating IoU        
        curr_int, curr_uni = returnInterUnion(labels_unhot, predictions)
        intersection += np.sum(curr_int)
        union += np.sum(curr_uni)        
        
        #Add to tensorboard
        writer.add_scalar('Iteration Validation Loss', loss.item(), 
                          epoch*len(validloader) + i + 1)
        
        # Computing Confusion matrix, leave out for now as it takes too long        
        y_pred = predictions.flatten().cpu().numpy()        
        y_true = labels_unhot.flatten().cpu().numpy()
        conf_matrix += confusion_matrix(y_true, y_pred, labels=class_list)
        
        #Calculate current precision/recall/f1 for CAFO class        
        precision, recall, f_score = returnPreReF(conf_matrix, 1)                
        
        #Update progress bar        
        avg_loss = acc_loss/(i + 1)                
        acc_avg = valid_correct/valid_denom
        desc = f'Epoch {epoch} - loss {avg_loss:.4f} - acc {acc_avg:.4f} - ' \
        f'Mean IoU {(intersection/union):.4f} - Precision {precision:.4f} - ' \
        f'Recall {recall:.4f}'
    
        epoch_pbar.set_description(desc)
        epoch_pbar.update(1)
        
            
    return valid_loss, valid_correct, conf_matrix, intersection/union
                        
def train(trainloader, validloader, model, device):
    """
    Parameters
    ----------
    trainloader : DataLoader object
        A dataloader for the training data.
    validloader : DataLoader object
        A dataloader for the validation data.
    model : TYPE
        The model used for training.
    device : TYPE
        Torch object indicating gpu/cpu.

    Returns
    -------
    None.

    """
    train_num_batches = len(trainloader)
    valid_num_batches = len(validloader)
    train_num_examples = len(trainloader.dataset)
    valid_num_examples = len(validloader.dataset)
    
    #Set model to either cpu or gpu
    model.to(device)            
    
    #Define loss function, weighted by class
    if args.dataset == "groundtruth":        
        pos_weight = torch.tensor([150, 300, 1])
        pos_weight = torch.reshape(pos_weight,(1,3,1,1)).to(device)
    elif args.dataset == "driven":
        pos_weight = torch.tensor([1, 10])
        pos_weight = torch.reshape(pos_weight,(1,2,1,1)).to(device)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
    
    #Set optimizers
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, 
                                     weight_decay = args.wd)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, 
                                     weight_decay = args.wd, 
                                     momentum = args.momentum)

    #Create Tensorboard    
    today = date.today()
    date_prefix = today.strftime("%m_%d")
    log_dir_suffix = f"{date_prefix}_{args.dataset}_lr_{args.lr}_epochs_{args.epochs}_batch_size_{args.batch_size}"
    log_dir = f"../logs/{args.dataset}/" + log_dir_suffix
    writer = SummaryWriter(log_dir=log_dir)
    
    best_loss = 1e9
    
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    # Reset gradients in model
    model.zero_grad()     
    
    for epoch in range(args.epochs):
        ### TRAINING ###
        print("Beginning Training in Epoch " + str(epoch))
        with tqdm(total = train_num_batches) as epoch_pbar:
            model.train()
            train_loss, train_correct, \
                train_IoU = train_one_epoch(epoch, train_num_batches, model, 
                                            device, trainloader, epoch_pbar, 
                                            optimizer, writer, criterion)
                        
        ### VALIDATION ###
        print("Beginning Validation in Epoch " + str(epoch))
        valid_loss = []
        valid_correct = 0
    
        if args.dataset == "driven":            
            conf_matrix = np.zeros((2,2))
            class_list = [0, 1]
        elif args.dataset == "groundtruth":
            conf_matrix = np.zeros((3,3))
            class_list = [0, 1, 2]
                
        with tqdm(total = valid_num_batches) as epoch_pbar:
            model.eval()                           
            valid_loss, valid_correct, \
                conf_matrix, valid_IoU = valid_one_epoch(epoch, valid_num_batches, model, 
                                                         device, validloader, epoch_pbar, 
                                                         optimizer, writer, criterion,
                                                         conf_matrix, class_list)
        
        # Calculate Precision, Recall, F-score for the CAFO class, which is 2
        if args.dataset == "driven":            
            precision, recall, f_score = returnPreReF(conf_matrix, 1)
        elif args.dataset == "groundtruth":
            precision, recall, f_score = returnPreReF(conf_matrix, 2)
                
        ### UPDATE TENSORBOARD ###
        writer.add_scalar('Epoch Training Loss', np.mean(train_loss), epoch)
        writer.add_scalar('Epoch Validation Loss', np.mean(valid_loss), epoch)
        writer.add_scalar('Epoch Training Accuracy', 
                          train_correct/train_num_examples, epoch)
        writer.add_scalar('Epoch Validation Accuracy', 
                          valid_correct/valid_num_examples, epoch)
        writer.add_scalar('Epoch Training Mean IoU', train_IoU, epoch)
        writer.add_scalar('Epoch Validation Mean IoU', valid_IoU, epoch)
        writer.add_scalar('Precision', precision, epoch)
        writer.add_scalar('Recall', recall, epoch)
        writer.add_scalar('F1 Score', f_score, epoch)        

        ### Save if Model gets best loss ###
        if args.save_model:
            if np.sum(valid_loss) < best_loss:
                best_loss = np.mean(valid_loss)
                torch.save(model.state_dict(), 
                           f"../../../saved_models/{args.dataset}/" + log_dir_suffix + ".pth")
        
def main():
    print(args)
    
    # Load Dataset and Dataloader
    if args.dataset == "driven":
        dataset = drivenDataset(args.image_path, args.mask_path)
        model = UNet(3, 2) # 3 Channels, 2 Classes (background, building)
    elif args.dataset == "groundtruth":    
        dataset = groundTruthDataset(args.filepath)
        #change once we have more models
        #if args.model == "unet":
        model = UNet(3, 3) # 3 Channels, 3 Classes (background, lagoon, CAFO)
    else:
        raise Exception("Invalid dataset provided")
        
    datasets = splitDataset(dataset)
    trainloader, validloader, testloader = returnLoaders(datasets, args.batch_size, args.shuffle)    
    
    # Set to GPU if it is there
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    #Train the model
    train(trainloader, validloader, model, device)
    
    
if __name__ == "__main__":
    main()