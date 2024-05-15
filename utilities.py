import matplotlib.pyplot as plt
import numpy as np
import random
from functools import reduce
import itertools
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from sklearn.model_selection import train_test_split



# class SimDataset(Dataset):
#     def __init__(self, A, B, transform=None):
#         self.before_images = A
#         self.after_images = B
#         self.transform = transform

#         # get random 

#     def __len__(self):
#         return len(self.before_images)

#     def __getitem__(self, idx):
#         image_before = self.input_images[idx]
#         image_after = self.target_masks[idx]
#         if self.transform:
#             image = self.transform(image)

#         return [image, mask]
    

def prepare_images(A, B):
    # normalize the data
    # concatenate every 2 images together in each set
    # create image tensors
    # use the same transformations for train/val in this

    # normalize imagee
    A = (np.array(A)/255.0).astype(np.float32)
    B = (np.array(B)/255.0).astype(np.float32)

    # concatenate every 2 images together in each set
    final_set = np.concatenate((A, B), axis=1)

    # create image tensors
    final_set = torch.tensor(final_set)

    return final_set



def get_data_loaders(A, B):
    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])

    # split the data into train, validation, and test
    # 80% train, , 10% test, 10% validation
    # First, split into training and temp sets (80% training, 20% temp)
    train_A, test_temp_A, train_B, test_temp_B = train_test_split(A, B, test_size=0.2, random_state=42)

    # Then split temp set into test and validation sets (10% test, 10% validation)
    test_A, val_A, test_B, val_B = train_test_split(test_temp_A, test_temp_B, test_size=0.5, random_state=42)

    # Prepare the images
    final_train_set = prepare_images(train_A, train_B)
    final_val_set = prepare_images(val_A, val_B)
    final_test_set = prepare_images(test_A, test_B)

    print("Train Set Size: ", final_train_set.shape)
    print("Validation Set Size: ", final_val_set.shape)
    print("Test Set Size: ", final_test_set.shape)

    image_datasets = {
        'train': final_train_set, 'val': final_val_set, 'test': final_test_set
    }

    batch_size = 25

    dataloaders = {
        'train': DataLoader(final_train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(final_val_set, batch_size=batch_size, shuffle=False, num_workers=0),
        'test': DataLoader(final_test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    }

    

    return dataloaders


def dice_loss(pred, target, smooth=1.):
    '''
     The Dice coefficient D between two sets 𝐴 and 𝐵 is defined as:
     D= (2×∣A∩B∣)/ (∣A∣+∣B∣)
     ∣A∩B∣: total no of pixels in pred,gold that has +ve
    '''
    pred = pred.contiguous() # contiguous() is a method that is used to ensure that the tensor is stored in a contiguous block of memory.
    target = target.contiguous()  #torch.Size([25, 6, 192, 192])

    intersection = (pred * target).sum(dim=2).sum(dim=2)  # Sumation of Both Width & Height

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def jaccard_index(pred, target, smooth=1.0):
    '''
    Jaccard Index (IoU) between two sets 𝐴 and 𝐵 is defined as:
    J(A, B) = 1 - (∣A∩B∣ / ∣A∪B∣)
    Where:
    ∣A∩B∣: Intersection of sets A and B
    ∣A∪B∣: Union of sets A and B
    '''
    pred = pred.contiguous() 
    target = target.contiguous() 

    intersection = (pred * target).sum(dim=2).sum(dim=2)  
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) - intersection

    IOU = ((intersection + smooth) / (union + smooth))
    
    return 1- IOU.mean()


def calc_loss(predictions, targets, metrics, bce_weight=0.5):
    # Binary Cress Entropy
    # In PyTorch, binary_cross_entropy_with_logits is a loss function that combines a sigmoid activation function and binary cross-entropy loss.
    # However, it doesn't explicitly apply the sigmoid function to the input. Instead, it expects the input to be logits, which are the raw outputs of a model without applying any activation function.
    for prediction, target in zip(predictions, targets):
        bce = F.binary_cross_entropy_with_logits(prediction, target)

        prediction = F.sigmoid(prediction)
        dice = dice_loss(prediction, target)

        # Custom Loss function that combines bce & dice losses
        # Binary Cross-Entropy (BCE) Loss: BCE loss aims to minimize the difference between the predicted probability distribution and the ground truth binary labels.
        # It penalizes deviations from the true binary labels, typically encouraging the model to output probabilities that align well with the ground truth.
        # Dice Loss: Dice loss aims to maximize the overlap between the predicted segmentation mask and the ground truth mask.
        # It penalizes deviations from the true segmentation mask, typically encouraging the model to produce segmentations that align well with the ground truth boundaries.
        loss = bce * bce_weight + dice * (1 - bce_weight)

        jac_index=jaccard_index(prediction, target)


        metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
        metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
        metrics['jaccrod_index']+=jac_index.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, A, B, optimizer, scheduler, bce_weight=0.5, num_epochs=25):
    dataloaders = get_data_loaders(A, B)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    loss_per_epoch=[]
    bce_loss_per_epoch=[]
    dice_loss_per_epoch=[]
    jacord_index_per_epoch=[]

    loss_per_val_epoch=[]
    bce_loss_per_val_epoch=[]
    dice_loss_per_val_epoch=[]
    jacord_index_per_val_epoch=[]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train by passing train condition to set_grad_enabled :D
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics,bce_weight=bce_weight)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()  # optimzation custom loss function :D
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'train':
              loss_per_epoch.append(epoch_loss)
              bce_loss_per_epoch.append(metrics['bce']/epoch_samples)
              dice_loss_per_epoch.append(metrics['dice']/epoch_samples)
              jacord_index_per_epoch.append(metrics['jaccrod_index']/epoch_samples)

            elif phase=="val":
              loss_per_val_epoch.append(epoch_loss)
              bce_loss_per_val_epoch.append(metrics['bce']/epoch_samples)
              dice_loss_per_val_epoch.append(metrics['dice']/epoch_samples)
              jacord_index_per_val_epoch.append(metrics['jaccrod_index']/epoch_samples)


            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, dataloaders['test'],loss_per_epoch,bce_loss_per_epoch,dice_loss_per_epoch,jacord_index_per_epoch,loss_per_val_epoch,bce_loss_per_val_epoch,dice_loss_per_val_epoch,jacord_index_per_val_epoch



def compute_test_loss(pred,target):
    pred = F.sigmoid(pred)
    # Dice Loss
    # dice_score = (1- dice_loss(pred, target)).cpu().numpy() * target.size(0)
    dice_score = (1- dice_loss(pred, target)).cpu().numpy()

    # jaccord Index
    jaccord_index= jaccard_index(pred, target).cpu().numpy()


    return dice_score,jaccord_index

def test_model(model, test_data_loader):
    print("Testing Model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_idx=0

    dice_score=0
    dice_loss=0
    jaccord_index=0

    for inputs, labels in test_data_loader:
        print(f'{batch_idx}/{len(test_data_loader)}')
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            dice_score_i,jaccord_index_i = compute_test_loss(outputs, labels)

            dice_score+=dice_score_i
            jaccord_index+=jaccord_index_i
            dice_loss+=(1-dice_score_i)
    
    # Average Dice Score
    dice_score = dice_score/len(test_data_loader)
    dice_loss = dice_loss/len(test_data_loader)
    jaccord_index = jaccord_index/len(test_data_loader)

    print(f'Test: DiceLoss : {dice_loss} Dice Score: {dice_score} Jaccord Index: {jaccord_index}') 
    return

def run(UNet, A, B, lr=1e-4, step_size=30, gamma=0.1, num_epochs=60, bce_weight=0.5, test=False):
    # random.seed(27)
    # np.random.seed(27)
    # torch.manual_seed(27)    
    # torch.cuda.manual_seed_all(27)
    num_class = 6

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(num_class).to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # The StepLR scheduler decreases the learning rate of the optimizer by a factor (gamma) at specified intervals (step_size).
    # Here, the learning rate will be decreased by a factor of 0.1 every 30 epochs.
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

    model, test_dataloader, loss_per_epoch, bce_loss_per_epoch, dice_loss_per_epoch, jacord_index_per_epoch, loss_per_val_epoch, bce_loss_per_val_epoch, dice_loss_per_val_epoch, jacord_index_per_val_epoch = train_model(model, A, B, optimizer_ft, exp_lr_scheduler, bce_weight=bce_weight, num_epochs=num_epochs)
    print("Done Training")


    # if test:
    #   print("Testing Model")
    #   model.eval()  # Set model to the evaluation mode
    #   trans = transforms.Compose([
    #       transforms.ToTensor(),
    #       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
    #   ])

    #   # Test DataSet
    #   test_model(model, test_dataloader)

      # Create another simulation dataset for test
    #   test_dataset = SimDataset(3, transform = trans)
    #   test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)

    #   # Get the first batch
    #   inputs, labels = next(iter(test_loader))
    #   inputs = inputs.to(device)
    #   labels = labels.to(device)

    #   # Predict
    #   pred = model(inputs)
    #   # The loss functions include the sigmoid function.
    #   pred = F.sigmoid(pred)
      # TODO: Computing Dice Score & Jaccard Index
      # Dice Score
    #   dice_sample_loss=dice_loss(pred, labels)
    #   dice_sample_score = 1-dice_sample_loss
    #   # jaccord Index
    #   jaccord_index= jaccard_index(pred, labels)
    #   print(f'Test(Sample): DiceLoss : {dice_sample_loss} Dice Score: {dice_sample_score} Jaccord Index: {jaccord_index}') 

    #   pred = pred.data.cpu().numpy()


    #   # Change channel-order and make 3 channels for matplot
    #   input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

    #   # Map each channel (i.e. class) to each color
    #   target_masks_rgb = [masks_to_colorimg(x) for x in labels.cpu().numpy()]
    #   pred_rgb = [masks_to_colorimg(x) for x in pred]

    #   plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])

    #   return
    return loss_per_epoch,bce_loss_per_epoch,dice_loss_per_epoch,jacord_index_per_epoch,loss_per_val_epoch,bce_loss_per_val_epoch,dice_loss_per_val_epoch,jacord_index_per_val_epoch