import os

# importing the libraries
import pandas as pd
import numpy as np
import pip
import json
import PIL.Image as imageLib
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, \
    Dropout
from torch.optim import Adam, SGD

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

preprocess = transforms.Compose([
    # resize
    transforms.Resize(256),
    # center-crop
    transforms.CenterCrop(256),
    # to-tensor
    transforms.ToTensor(),
    # normalize
])


class InputData:
    def __init__(self, image, jsonfile):
        self.image = image
        self.json = jsonfile


class PhotoData(Dataset):
    def __init__(self, path_to_imgs, path_to_json):
        super(PhotoData, self).__init__()
        self.path_to_imgs = path_to_imgs
        self.path_to_json = path_to_json
        self.image_ids = os.listdir(path_to_imgs)
        self.data_list = []
        for i in range(self.image_ids.__len__()):
            image, json = self.loaditem(i)
            x = InputData(image, json)
            self.data_list.append(x)

    def __getitem__(self, index):
        return self.data_list[index]

    def loaditem(self, idx):
        img_id = self.image_ids[idx]
        print(img_id)
        img = imageLib.open(os.path.join(self.path_to_imgs, img_id))
        y = preprocess(img)
        img = y
        # img = plt.imread(os.path.join(self.path_to_imgs, img_id))
        json_id = img_id.replace('png', 'json')
        print(json_id)
        # json = pd.read_json(os.path.join(self.path_to_json, json_id))
        with open(self.path_to_json + json_id) as file:
            json_file = json.load(file)
        return img, json_file

    def __len__(self):
        return len(self.image_ids)


l_data = PhotoData(
    path_to_imgs='C:/UnityProjects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/Training Images/Images/',
    path_to_json='C:/UnityProjects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/Training Images/Parameters/')

reverse_preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        np.array,
    ]
)
plt.show()

# Example of unpacking json data
# with open(
#      "D:/Unity Projects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/Training Images\Parameters/0.json") as f:
#    sample_submission = json.load(f)
print(l_data.__getitem__(8).json[5])
print(len(l_data.__getitem__(8).json))
# training_Data = DataLoader(l_data,batch_size=20,shuffle)


class LeNet5(Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.c2 = Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.c3 = Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.max_pool = MaxPool2d(kernel_size=2, stride=2)
        self.relu = ReLU()
        self.fc1 = Linear(in_features=120, out_features=84)
        self.fc2 = Linear(in_features=84, out_features=10)

    def forward(self, img):
        x = self.c1(img)
        x = self.relu(self.max_pool(x))
        x = self.c2(x)
        x = self.relu(self.max_pool(x))
        x = self.relu(self.c3(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Model:
    def __init__(self, model, learning_rate, device):
        self.model = model
        self.lr = learning_rate
        self.loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.device = device

    def batch_accuracy(self, output, target):
        # output shape: [batch, 10]
        output = nn.functional.softmax(output, dim=1)
        output = output.argmax(1)
        acc = torch.sum(output == target) / output.shape[0]
        return acc.cpu() * 100

    def train_step(self, dataset):
        self.model.train()
        batch_loss = []
        batch_acc = []
        for batch in dataset:
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            self.opt.zero_grad()

            outputs = self.model(inputs)

            loss = self.loss(outputs, targets)
            loss.backward()
            self.opt.step()
            batch_loss.append(loss.item())
            batch_acc.append(self.batch_accuracy(outputs, targets))

        self.train_loss.append(np.mean(batch_loss))
        self.train_acc.append(np.mean(batch_acc))

    def validation_step(self, dataset):
        self.model.eval()
        batch_loss = []
        batch_acc = []
        with torch.no_grad():
            for batch in dataset:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)

                outputs = self.model(inputs)

                loss = self.loss(outputs, targets)
                batch_loss.append(loss.item())
                batch_acc.append(self.batch_accuracy(outputs, targets))

        self.val_loss.append(np.mean(batch_loss))
        self.val_acc.append(np.mean(batch_acc))

    def test_step(self, dataset):
        self.model.eval()
        batch_acc = []
        with torch.no_grad():
            for batch in dataset:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)

                outputs = self.model(inputs)
                batch_acc.append(self.batch_accuracy(outputs, targets))

        print("Accuracy : ", np.mean(batch_acc), "%")


epoch = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-4
batch = 32
lenet5 = LeNet5().to(device)

#
#
# class Net(Module):
#     def __init__(self):
#         super(Net, self).__init__()
#
#         self.cnn_layers = Sequential(
#             # Defining a 2D convolution layer
#             Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(4),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
#             # Defining another 2D convolution layer
#             Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(4),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
#         )
#
#         self.linear_layers = Sequential(
#             Linear(4 * 7 * 7, 10)
#         )
#
#     # Defining the forward pass
#     def forward(self, x):
#         x = self.cnn_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear_layers(x)
#         return x
#
# model = Net()
#
#
# optimizer = Adam(model.parameters(), lr=0.07)
# # defining the loss function
# criterion = CrossEntropyLoss()
# # checking if GPU is available
# if torch.cuda.is_available():
#     model = model.cuda()
#     criterion = criterion.cuda()
#
# print(model)
# #Import Data
# data_dir = "D:/Unity Projects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/Training Images/Images"
#
# dataset = ImageFolder(data_dir, transform)
#
# img, label = dataset[0]
# print(img.shape, label)
#
# # defining the number of epochs
# n_epochs = 25
# # empty list to store training losses
# train_losses = []
# # empty list to store validation losses
# val_losses = []
#
#
#
#
# # training the model
# #for epoch in range(n_epochs):
# #   train(epoch)
#
#
# # plotting the training and validation loss
# #plt.plot(train_losses, label='Training loss')
# #plt.plot(val_losses, label='Validation loss')
# #plt.legend()
# #plt.show()
