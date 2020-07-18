# Author: Louis Chen
# Standard library
import copy
import glob
import multiprocessing
import os
import time

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# Related third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io, transform
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm




# train data file looks './train/dog.10435.jpg'
# test data file looks './test/10435.jpg'
def extract_class_from(path):
    file = path.split('/')[-1]
    return file.split('.')[0]

'''The train_model function handles the training and validation of a given model. As input, 
it takes a PyTorch model, a dictionary of dataloaders, a loss function, an optimizer, a specified number 
of epochs to train and validate for, and a boolean flag for when the model is an Inception model. The 
is_inception flag is used to accomodate the Inception v3 model, as that architecture uses an auxiliary 
output and the overall model loss respects both the auxiliary output and the final output, as described 
here. The function trains for the specified number of epochs and after each epoch runs a full validation 
step. It also keeps track of the best performing model (in terms of validation accuracy), and at the end 
of training returns the best performing model. After each epoch, the training and validation accuracies 
are printed.'''
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = {'accuracy': [],
               'val_accuracy': [],
               'loss': [],
               'val_loss': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                history['accuracy'].append(epoch_acc.item())
                history['loss'].append(epoch_loss)
            else:
                history['val_accuracy'].append(epoch_acc.item())
                history['val_loss'].append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history




'''dataset class'''


class DogVsCatDataset(Dataset):

    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.file_list[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        label_category = extract_class_from(img_name)
        label = 1 if label_category == 'dog' else 0

        return image, label






def main():
    all_train_files = glob.glob(os.path.join(train_dir, '*.jpg'))
    train_list, val_list = train_test_split(all_train_files, random_state=42)
    print(len(train_list))
    print(len(val_list))
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    # Create training and validation datasets
    image_datasets = {
        'train': DogVsCatDataset(train_list,
                                 transform=data_transforms['train']),
        'val': DogVsCatDataset(val_list,
                               transform=data_transforms['val'])
    }

    # Create training and validation dataloaders
    dataloaders_dict = {x: DataLoader(image_datasets[x],
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.vgg16(pretrained=True)
    model_ft.classifier[6] = nn.Linear(4096, num_classes)

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

    test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
    test_data_transform = data_transforms['val']

    ids = []
    labels = []

    with torch.no_grad():
        for test_path in tqdm(test_list):
            img = Image.open(test_path)
            img = test_data_transform(img)
            img = img.unsqueeze(0)
            img = img.to(device)

            model_ft.eval()
            outputs = model_ft(img)
            preds = F.softmax(outputs, dim=1)[:, 1].tolist()

            test_id = extract_class_from(test_path)
            ids.append(int(test_id))
            labels.append(preds[0])

    output = pd.DataFrame({'id': ids,
                           'label': np.round(labels)})

    output.sort_values(by='id', inplace=True)
    output.reset_index(drop=True, inplace=True)

    output.to_csv('submission.csv', index=False)







if __name__ == '__main__':

    base_dir = 'C:/Users/Louis/Desktop/dogs-vs-cats-redux-kernels-edition'
    train_dir = 'C:/Users/Louis/Desktop/dogs-vs-cats-redux-kernels-edition/train'
    test_dir = 'C:/Users/Louis/Desktop/dogs-vs-cats-redux-kernels-edition/test'
    input_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Number of classes in the dataset
    num_classes = 2  # dog, cat

    # Batch size for training (change depending on how much memory you have)
    batch_size = 16

    # Number of epochs to train for
    num_epochs = 2

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True

    # Switch to perform multi-process data loading
    num_workers = multiprocessing.cpu_count()
    main()