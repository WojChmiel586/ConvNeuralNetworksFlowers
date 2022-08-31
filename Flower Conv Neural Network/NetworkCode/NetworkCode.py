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
        self.start_pos_x = round(jsonfile.get('startPos').get('x'), 3)
        self.start_pos_y = round(jsonfile.get('startPos').get('y'), 3)
        self.start_pos_z = round(jsonfile.get('startPos').get('z'), 3)
        self.tangent1_x = round(jsonfile.get('tangent1').get('x'), 3)
        self.tangent1_y = round(jsonfile.get('tangent1').get('y'), 3)
        self.tangent1_z = round(jsonfile.get('tangent1').get('z'), 3)
        self.tangent2_x = round(jsonfile.get('tangent2').get('x'), 3)
        self.tangent2_y = round(jsonfile.get('tangent2').get('y'), 3)
        self.tangent2_z = round(jsonfile.get('tangent2').get('z'), 3)
        self.end_pos_x = round(jsonfile.get('endPos').get('x'), 3)
        self.end_pos_y = round(jsonfile.get('endPos').get('y'), 3)
        self.end_pos_z = round(jsonfile.get('endPos').get('z'), 3)
        self.edge_ring_count = jsonfile.get('_edgeRingCount')
        self.stem_radius = jsonfile.get('_stemRadius')
        self.cylinder_vertex_count = jsonfile.get('_cylinderVertexCount')
        self.petal_colour_r = jsonfile.get('_petalColour').get('r')
        self.petal_colour_g = jsonfile.get('_petalColour').get('g')
        self.petal_colour_b = jsonfile.get('_petalColour').get('b')
        self.radius = jsonfile.get('_radius')
        self.vertical_lines = jsonfile.get('_verticalLines')
        self.horizontal_lines = jsonfile.get('_horizontalLines')
        self.json = jsonfile


class PhotoData(Dataset):
    def __init__(self, path_to_imgs, path_to_json):
        super(PhotoData, self).__init__()
        self.path_to_imgs = path_to_imgs
        self.path_to_json = path_to_json
        self.image_ids = os.listdir(path_to_imgs)
        self.data_list = []
        for i in range(self.image_ids.__len__()):
            image, json = self.load_item(i)
            x = InputData(image, json)
            self.data_list.append(x)

    def __getitem__(self, index):
        return self.data_list[index]

    def load_item(self, idx):
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


images_path_home = 'D:/Unity Projects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/Training Images/Images/'
json_path_home = 'D:/Unity Projects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/Training Images/Parameters/'
path_images_laptop = 'C:/UnityProjects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/Training Images/Images/'
path_json_laptop = 'C:/UnityProjects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/Training Images/Parameters/'

l_data = PhotoData(
    path_to_imgs=images_path_home,
    path_to_json=json_path_home)

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
print(l_data.__getitem__(8).json.get('_petalColour'))
print(l_data.__getitem__(8).petal_colour_g)
print(len(l_data.__getitem__(8).json))


# training_Data = DataLoader(l_data,batch_size=20,shuffle)


class LeNet5(Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.c1 = Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        # 256-5 +1 =252
        self.c2 = Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.c3 = Conv2d(in_channels=16, out_channels=64, kernel_size=4, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = ReLU()
        self.fc1 = Linear(in_features=64 * 29 * 29, out_features=120)
        self.fc2 = Linear(in_features=120, out_features=84)
        self.fc3 = Linear(in_features=84, out_features=21)

    def forward(self, img):
        x = self.c1(img)
        # 252
        x = self.relu(self.avg_pool(x))
        # 126
        x = self.c2(x)
        # 126-5 +1 = 122
        x = self.relu(self.avg_pool(x))
        # 61
        x = self.c3(x)
        # 61 -4 + 1 = 58
        x = self.relu(self.avg_pool(x))
        # 29
        x = x.view(-1, 64 * 29 * 29)
        # x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Model:
    def __init__(self, model, learning_rate, device):
        self.model = model
        self.lr = learning_rate
        self.loss = nn.MSELoss()
        self.opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
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
