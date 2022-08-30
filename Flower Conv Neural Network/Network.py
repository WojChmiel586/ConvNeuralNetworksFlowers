import os

# importing the libraries
import pandas as pd
import numpy as np
import pip
import json

import torch
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


class InputData():
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
        img = plt.imread(os.path.join(self.path_to_imgs, img_id))
        img2 = transforms.Resize(32)
        json_id = img_id.replace('png', 'json')
        print(json_id)
        # json = pd.read_json(os.path.join(self.path_to_json, json_id))
        with open(self.path_to_json + json_id) as file:
            jsonFile = json.load(file)
        return img, jsonFile

    def __len__(self):
        return len(self.image_ids)


l_data = PhotoData(
    path_to_imgs='D:/Unity Projects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/Training Images/Images',
    path_to_json='D:/Unity Projects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/Training Images/Parameters/')
# print(l_data.data_list.__len__())
# print(l_data.__getitem__(15).image)
print(l_data.__getitem__(8).json)
print(l_data.__getitem__(8).image)


with open(
        "D:/Unity Projects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/Training Images\Parameters/0.json") as f:
    sample_submission = json.load(f)

print(l_data.__getitem__(8).json['_horizontalLines'])

# transform = transforms.Compose([
#     # resize
#     transforms.Resize(32),
#     # center-crop
#     transforms.CenterCrop(32),
#     # to-tensor
#     transforms.ToTensor(),
#     # normalize
#     #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])
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
