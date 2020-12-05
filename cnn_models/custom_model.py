from CustomDataLoader import CustomDataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader


class unet_contract_double(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, padding: int):
        super(unet_contract_double, self).__init__()
        self.first_conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding)
        self.second_conv = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.batch_norm(self.first_conv(x)))
        x = F.relu(self.batch_norm(self.second_conv(x)))

        return x


class unet_expand_double(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, padding: int):
        super(unet_expand_double, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.first_conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding)
        self.second_conv = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.up_sample(x)
        x = F.relu(self.batch_norm(self.first_conv(x)))
        x = F.relu(self.batch_norm(self.second_conv(x)))

        return x


class unet_classification(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, kernel_size: tuple):
        super(unet_classification, self).__init__()
        self.out_conv = nn.Conv2d(in_channels, n_classes, kernel_size)

    def forward(self, x):
        x = self.out_conv(x)
        return x


class UNet(nn.Module):

    def __init__(self, out_classes: int):
        super(UNet, self).__init__()
        self.out_classes = out_classes
        self.pool = nn.MaxPool2d(2)

        self.contract_1 = unet_contract_double(1, 64, (3,3), 1)
        self.contract_2 = unet_contract_double(64, 128, (3,3), 1)
        self.contract_3 = unet_contract_double(128, 256, (3,3), 1)
        self.contract_4 = unet_contract_double(256, 512, (3,3), 1)
        self.contract_5 = unet_contract_double(512, 1024, (3,3), 1)

        self.expand_1 = unet_expand_double(1024, 512, (3,3), 1)
        self.expand_2 = unet_expand_double(512, 256, (3,3), 1)
        self.expand_3 = unet_expand_double(256, 128, (3,3), 1)
        self.expand_4 = unet_expand_double(128, 64, (3,3), 1)

        self.classification = unet_classification(64, self.out_classes, (1,1))

    def forward(self, x):
        # the whole architecture here
        # Contraction
        x1 = self.contract_1(x)
        x1 = self.pool(x1)

        x2 = self.contract_2(x1)
        x2 = self.pool(x2)

        x3 = self.contract_3(x2)
        x3 = self.pool(x3)

        x4 = self.contract_4(x3)
        x4 = self.pool(x4)

        x5 = self.contract_5(x4)
        x5 = self.pool(x5)

        # Expansion
        x6 = self.expand_1(x5)
        x7 = self.expand_2(x6)
        x8 = self.expand_3(x7)
        x9 = self.expand_4(x8)

        # Contract again
        x10 = self.pool(x9)
        x11 = self.pool(x10)
        x12 = self.pool(x11)
        x13 = self.pool(x12)
        x14 = self.pool(x13)
        x15 = self.pool(x14)

        # Output
        output = self.classification(x15)
        output = output.squeeze()
        return output

par = [[0.1,0.9],[0.05,0.9],[0.02,0.9],[0.01,0.9], [0.1,0.95],[0.05,0.95],[0.02,0.95],[0.01,0.95], [0.001,0.95]]

for p in par:

    lr = p[0]
    momentum = p[1]

    print(f'LR is {lr} and momentum is {momentum}')

    u_net_model = UNet(12)
    print(u_net_model)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])

    train_dataset = CustomDataLoader('../images/train', '../image_processing/train_all.csv', transform=preprocess)
    validation_dataset = CustomDataLoader('../images/train', '../image_processing/valid_all.csv', transform=preprocess)

    # Parameters for data loader
    params = {'batch_size': 64,  # 64
              'shuffle': False,
              'num_workers': 6}
    max_epochs = 10

    train_loader = DataLoader(train_dataset, **params)
    valid_loader = DataLoader(validation_dataset, **params)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    u_net_model.to(device)
    print(f'Running with device {device}')

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(u_net_model.parameters(), lr=lr, momentum=momentum)

    # Train the network
    for epoch in range(max_epochs):

        print('Epoch {}/{}'.format(epoch, max_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:

            # Training
            if phase == 'train':
                u_net_model.train()  # Set model to training mode
            else:
                u_net_model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0


            for batch, labels in train_loader if phase == 'train' else valid_loader:
                # Transfer to GPU
                batch, labels = batch.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = u_net_model(batch)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels.type(torch.int64))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * batch.size(0)
                running_corrects += torch.sum(preds == labels.data)

            num_samples = len(train_dataset) if phase == 'train' else len(validation_dataset)
            epoch_loss = running_loss / num_samples
            epoch_acc = running_corrects.double() / num_samples

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


    print('Finished Training')