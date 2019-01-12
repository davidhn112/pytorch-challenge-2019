# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 12:46:32 2018
PyTorch Scholarship Challenge 2018-2019 Final Lab
Flower Classification, 102 Classes
https://www.udacity.com/facebook-pytorch-scholarship

@author: David Nguyen
http://www.thedavidnguyenblog.xyz
https://www.linkedin.com/in/david-nguyen-7abb8352/
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import datetime
import copy
import json

plt.ion()   # interactive mode

# suggested for num workers > 0 on windows
if __name__ == "__main__":

    # date timestamp for filenaming checkpoint
    date = datetime.datetime.now()
    formatted_date = date.strftime('%Y%m%d_%H%M')

    # check if CUDA is available
    data_parallel = False
    train_on_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if train_on_gpu else "cpu")

    if train_on_gpu:
        print('CUDA is available!  Training on GPU ...')

        # if multiple GPUs are available, wrap model in DataParallel
        if torch.cuda.device_count() > 1:
            data_parallel = True

        # print memory allocation for GPU
        for i in range(torch.cuda.device_count()):
            print('GPU {}  :  {:.2f}  GB'.format(i,
                  torch.cuda.memory_allocated(i)/1000000000))
    else:
        print('CUDA is not available.  Training on CPU only...')

    # data repository
    train_dir = 'flower_data/train'
    valid_dir = 'flower_data/valid'
    test_dir = 'flower_data/test'

    # hyperparameters
    num_workers = 16  # number of subprocesses to use for data loading
    batch_size = 128  # how many samples per batch to load
    lr = 0.0001  # learning rate

    # transforms for the training and validation sets
    data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])]),
    }

    # load training and validation sets
    training_dataset = datasets.ImageFolder(train_dir,
                                            transform=data_transforms['train'])
    training_dataloader = torch.utils.data.DataLoader(training_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=num_workers)

    valid_dataset = datasets.ImageFolder(valid_dir,
                                         transform=data_transforms['val'])
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=num_workers)

    test_dataset = datasets.ImageFolder(test_dir,
                                        transform=data_transforms['val'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers)

    # map the image classes to directory labels
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    number_of_classes = len(cat_to_name)
    class_names = training_dataset.classes

    # function to visualize tensor
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    # model training definition
    def train_model(model, criterion, optimizer, scheduler, num_epochs=1):
        # record total processing time
        since = time.time()

        # save stats to report at end of training
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_val_epoch = int(0)
        best_test_acc_model_wts = copy.deepcopy(model.state_dict())
        best_test_acc = 0.0
        best_test_epoch = int(0)

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            # each epoch has a training, validation and test phase
            for phase in ['Train', 'Validation', 'Test']:
                if phase == 'Train':
                    # scheduler.step()
                    model.train()  # Set model to training mode
                    dataloader = training_dataloader
                    dataset_size = len(training_dataset)
                if phase == 'Validation':
                    model.eval()   # Set model to evaluate mode
                    dataloader = valid_dataloader
                    dataset_size = len(valid_dataset)
                if phase == 'Test':
                    model.eval()   # Set model to evaluate mode
                    dataloader = test_dataloader
                    dataset_size = len(test_dataset)

                running_loss = 0.0
                running_corrects = 0

                # iterate over data
                for inputs, labels in dataloader:
                    # send tensors to CUDA if available
                    if train_on_gpu:
                        inputs, labels = inputs.to(device), labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward pass
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'Train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward prop + optimize only if in training phase
                        if phase == 'Train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_size
                epoch_acc = running_corrects.double() / dataset_size
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # save best validation model and epoch
                if phase == 'Validation':
                    scheduler.step(running_loss)

                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        print('Saving Weights')
                        best_val_epoch = epoch

                # save best accuracy model and epoch
                if phase == 'Test' and epoch_acc > best_test_acc:
                    best_test_acc = epoch_acc
                    best_test_acc_model_wts = copy.deepcopy(model.state_dict())
                    print('Saving Weights')
                    best_test_epoch = epoch

            print()

        # print final stats to console
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
              time_elapsed % 60))
        print('Best Validation Acc: {:4f}'.format(best_acc))
        print('Best Val Epoch: {:.0f}'.format(best_val_epoch+1))
        print('Best Test Dataset Acc: {:4f}'.format(best_test_acc))
        print('Best Test Epoch: {:.0f}'.format(best_test_epoch+1))

        # load best model weights for validation set and save checkpoint
        model.load_state_dict(best_model_wts)
        if data_parallel:
            torch.save(model.module.state_dict(),
                       str('densenet161_{}Epoch_Best_Validation_{}.pt'.format(
                               num_epochs, formatted_date)))
        else:
            torch.save(model.state_dict(),
                       str('densenet161_{}Epoch_Best_Validation_{}.pt'.format(
                               num_epochs, formatted_date)))

        # load best model weights for test set and save checkpoint
        model.load_state_dict(best_test_acc_model_wts)
        if data_parallel:
            torch.save(model.module.state_dict(),
                       str('densenet161_{}Epoch_Best_Acc_{}.pt'.format(
                               num_epochs, formatted_date)))
        else:
            torch.save(model.state_dict(),
                       str('densenet161_{}Epoch_Best_Acc_{}.pt'.format(
                               num_epochs, formatted_date)))

        return model

    # initialize pretrained model using densenet161
    model = models.densenet161(pretrained=True)

    # freeze all parameters since we're only optimizing final connected layer
    for param in model.parameters():
        param.requires_grad_(False)

    # create and replace last layer of pretrained model
    # parameters of new modules have requires_grad=True by default
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, number_of_classes)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer function
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr,
                           betas=(0.9, 0.999), eps=1e-8)

    # scheduler function
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                               factor=0.1, patience=5,
                                               verbose=False, threshold=0.0001,
                                               threshold_mode='rel',
                                               cooldown=0, min_lr=0, eps=1e-08)

    # move the model to multiple GPU, if available
    if train_on_gpu:
        if data_parallel:
            model = nn.DataParallel(model)
        model.to(device)

    # train
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=200)
