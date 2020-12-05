import csv
import numpy as np
import pandas as pd
import json
import os
import cv2

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy

from CustomDataLoader import CustomDataLoader

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

"""Use these parameters for training. Change as req'd"""
par = [[0.05, 0.2]]
batch_size = 64

for p in par:

    train_dataset = CustomDataLoader('../images/train', '../image_processing/train_all.csv', transform=preprocess)
    validation_dataset = CustomDataLoader('../images/train', '../image_processing/valid_all.csv', transform=preprocess)

    # Parameters
    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 6}
    max_epochs = 24

    # Declare data loaders
    train_loader = DataLoader(train_dataset, **params)
    valid_loader = DataLoader(validation_dataset, **params)

    # check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Running with device {device}')

    vgg16 = models.vgg16(pretrained=True)
    vgg16.to(device)

    # change the number of classes
    vgg16.classifier[6].out_features = 11
    print(vgg16)

    # freeze convolution weights
    for param in vgg16.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()

    lr = p[0]
    momentum = p[1]
    optimizer = optim.SGD(vgg16.parameters(), lr=lr, momentum=momentum)
    print(f'LR is {lr} and momentum is {momentum} batch size is {batch_size}')

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(vgg16.state_dict())
    best_acc = 0.0

    # Blank df to save results
    results = np.zeros(shape=(max_epochs, 4)) # train_loss, train_acc, val_loss, val_acc

    # Loop over epochs
    for epoch in range(max_epochs):
        print('Epoch {}/{}'.format(epoch, max_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:

            # Training
            if phase == 'train':
                vgg16.train()  # Set model to training mode
            else:
                vgg16.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for local_batch, local_labels in train_loader if phase == 'train' else valid_loader:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = vgg16(local_batch)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, local_labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * local_batch.size(0)
                running_corrects += torch.sum(preds == local_labels.data)

            if phase == 'train':
                exp_lr_scheduler.step()

            num_samples = len(train_dataset) if phase == 'train' else len(validation_dataset)
            epoch_loss = running_loss / num_samples
            epoch_acc = running_corrects.double() / num_samples

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            loss_col = 0 if phase == 'train' else 2
            acc_col = 1 if phase == 'train' else 3

            results[epoch, loss_col] = epoch_loss
            results[epoch, acc_col] = epoch_acc

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(vgg16.state_dict())

            print('-'*20)

        # Save results
        results_df = pd.DataFrame(results, columns=['train loss', 'train acc', 'val loss', 'val acc'])
        results_df.to_csv(f'results_lr_{lr}_mom_{momentum}.csv')

        # Save model and run on test dataset
        PATH = './optimized_vgg16_net.pth'
        torch.save(best_model_wts, PATH)


    print('EVALUATION ON TEST DATA')
    print('-'*23)
    PATH = './optimized_vgg16_net.pth'
    params = {'batch_size': 64,  # 64
              'shuffle': False,
              'num_workers': 6}

    test_dataset = CustomDataLoader('../images/test', '../image_processing/test_all.csv', transform=preprocess)
    test_loader = DataLoader(test_dataset, **params)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Running with device {device}')

    vgg16 = models.vgg16()
    if torch.cuda.is_available():
        vgg16.cuda()
    vgg16.classifier[6].out_features = 11
    vgg16.load_state_dict(torch.load(PATH))
    vgg16.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = vgg16(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the  test images: %d %%' % (100 * correct / total))


    # GET GROUP ACCURACY ON TEST DATA
    test_dataset = CustomDataLoader('../images/test', '../image_processing/test_all.csv', transform=preprocess)
    test_loader = DataLoader(test_dataset, **params)
    class_correct = list(0. for i in range(12))
    class_total = list(0. for i in range(12))
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = vgg16(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(64):
                try:
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                except:
                    print(labels)
                    print(c)

    for i in range(12):
        if class_total[i] != 0:
            print('Accuracy of %1d : %2d %%' % (i, 100 * class_correct[i] / class_total[i]))