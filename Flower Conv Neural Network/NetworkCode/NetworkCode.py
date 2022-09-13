import codecs
import math
import os

import numpy
# importing the libraries
import numpy as np
import pip
import json
import PIL.Image as imageLib
# import cv2
import sklearn

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, \
    Dropout
from torch.optim import Adam, SGD

import torchvision
import matplotlib.pyplot as plt
import torchvision
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
torchvision.models.
reverse_preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        np.array,
    ]
)


def NormaliseData(data, data_min, data_max):
    normal_val = (data - data_min) / (data_max - data_min)
    return normal_val


def ReverseNormalise(normalised_data, data_min, data_max):
    normal_val = normalised_data * (data_max - data_min) + data_min
    return normal_val


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_output_json(parameters, idx):
    with open(f"Results/{idx}results.json", "w") as write_file:
        json.dump(parameters, write_file, cls=NumpyArrayEncoder)


class InputData:
    def __init__(self, image, jsonfile):
        self.image = image
        # self.start_pos_x = round(jsonfile.get('startPos').get('x'), 3)
        # self.start_pos_y = round(jsonfile.get('startPos').get('y'), 3)
        # self.start_pos_z = round(jsonfile.get('startPos').get('z'), 3)
        # self.tangent1_x = round(jsonfile.get('tangent1').get('x'), 3)
        # self.tangent1_y = round(jsonfile.get('tangent1').get('y'), 3)
        # self.tangent1_z = round(jsonfile.get('tangent1').get('z'), 3)
        # self.tangent2_x = round(jsonfile.get('tangent2').get('x'), 3)
        # self.tangent2_y = round(jsonfile.get('tangent2').get('y'), 3)
        # self.tangent2_z = round(jsonfile.get('tangent2').get('z'), 3)
        # self.end_pos_x = round(jsonfile.get('endPos').get('x'), 3)
        # self.end_pos_y = round(jsonfile.get('endPos').get('y'), 3)
        # self.end_pos_z = round(jsonfile.get('endPos').get('z'), 3)
        # self.edge_ring_count = jsonfile.get('_edgeRingCount')
        # self.stem_radius = jsonfile.get('_stemRadius')
        # self.cylinder_vertex_count = jsonfile.get('_cylinderVertexCount')
        # self.petal_colour_r = round(jsonfile.get('_petalColour').get('r'), 3)
        # self.petal_colour_g = round(jsonfile.get('_petalColour').get('g'), 3)
        # self.petal_colour_b = round(jsonfile.get('_petalColour').get('b'), 3)
        # self.radius = jsonfile.get('_radius')
        # self.vertical_lines = jsonfile.get('_verticalLines')
        # self.horizontal_lines = jsonfile.get('_horizontalLines')
        self.parameters = np.float32(np.array(
            [  # round(jsonfile.get('startPos').get('x'), 3),
                # round(jsonfile.get('startPos').get('y'), 3),
                # round(jsonfile.get('startPos').get('z'), 3),
                round(jsonfile.get('tangent1').get('x') - jsonfile.get('startPos').get('x'), 3),
                round(jsonfile.get('tangent1').get('y') - jsonfile.get('startPos').get('y'), 3),
                round(jsonfile.get('tangent1').get('z') - jsonfile.get('startPos').get('z'), 3),
                round(jsonfile.get('tangent2').get('x') - jsonfile.get('tangent1').get('x'), 3),
                round(jsonfile.get('tangent2').get('y') - jsonfile.get('tangent1').get('y'), 3),
                round(jsonfile.get('tangent2').get('z') - jsonfile.get('tangent1').get('z'), 3),
                round(jsonfile.get('endPos').get('x') - jsonfile.get('tangent2').get('x'), 3),
                round(jsonfile.get('endPos').get('y') - jsonfile.get('tangent2').get('y'), 3),
                round(jsonfile.get('endPos').get('z') - jsonfile.get('tangent2').get('z'), 3),
                round(jsonfile.get('_petalColour').get('r')),
                round(jsonfile.get('_petalColour').get('g')),
                round(jsonfile.get('_petalColour').get('b'))]
        ))

        # m self.parameters = torch.from_numpy(parameters)


class PhotoData(Dataset):
    def __init__(self, path_to_imgs, path_to_json):
        super(PhotoData, self).__init__()
        self.path_to_imgs = path_to_imgs
        self.path_to_json = path_to_json
        self.image_ids = os.listdir(path_to_imgs)
        self.data_list = []
        self.average_flower = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        self.parameter_sum = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        self.parameters_min_max = np.array(
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            dtype=float)
        for i in range(self.image_ids.__len__()):
            image, json = self.load_item(i)
            x = InputData(image, json)
            self.parameter_sum = np.add(self.parameter_sum, x.parameters)
            self.data_list.append(x)
        self.average_flower = self.calculate_average()
        for j in range(self.data_list.__len__()):
                                                                                                            # ADD THE AVERAGE FLOWER BACK IN LATER
            self.data_list[j].parameters -= self.average_flower
            self.check_min_max(self.data_list[j].parameters)
        for k in range(self.data_list.__len__()):
            data_to_normalise = self.data_list[k].parameters
            for l in range(data_to_normalise.size - 3):
                normalised_data = NormaliseData(data_to_normalise[l], self.parameters_min_max[l, 0],
                                                self.parameters_min_max[l, 1])
                self.data_list[k].parameters[l] = normalised_data

    def __getitem__(self, index):
        return self.data_list[index].image, torch.from_numpy(self.data_list[index].parameters)

    def load_item(self, idx):
        img_id = self.image_ids[idx]
        print(img_id)
        img = imageLib.open(os.path.join(self.path_to_imgs, img_id))
        y = preprocess(img)
        img = y
        json_id = img_id.replace('png', 'json')
        print(json_id)
        with open(self.path_to_json + json_id) as file:
            json_file = json.load(file)
        return img, json_file

    def check_min_max(self, parameters):
        for i in range(parameters.size):
            # print("PARAMETER COMPARISON ", parameters[i], " CURRENT MIN ", self.parameters_min_max[i, 0])
            if parameters[i] < self.parameters_min_max[i, 0]:
                self.parameters_min_max[i, 0] = parameters[i]
            # print("PARAMETER COMPARISON ", parameters[i], " CURRENT MAX ", self.parameters_min_max[i, 1])
            if parameters[i] > self.parameters_min_max[i, 1]:
                self.parameters_min_max[i, 1] = parameters[i]

    # print("THESE ARE MIN MAX PARAMETERS",self.parameters_min_max)

    def calculate_average(self):
        amount = self.data_list.__len__()
        average = np.array(
            [amount, amount, amount, amount, amount, amount, amount, amount, amount, amount, amount, amount])
        self.parameter_sum[9] = 0
        self.parameter_sum[10] = 0
        self.parameter_sum[11] = 0
        result = self.parameter_sum / average
        return result

    def __len__(self):
        return len(self.data_list)


images_path_home = 'D:/Unity Projects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/Training Images/Images/'
json_path_home = 'D:/Unity Projects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/Training Images/Parameters/'
path_images_laptop = 'C:/UnityProjects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/Training Images/Images/'
path_json_laptop = 'C:/UnityProjects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/Training Images/Parameters/'


# Example of unpacking json data
# with open(
#      "D:/Unity Projects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/Training Images\Parameters/0.json") as f:
#    sample_submission = json.load(f)

# print(training_set.__getitem__(8).petal_colour_g)


class LeNet5(Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        torch.nn.init.xavier_uniform_(self.c1.weight)
        self.c2 = Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        torch.nn.init.xavier_uniform_(self.c2.weight)
        self.c3 = Conv2d(in_channels=16, out_channels=64, kernel_size=4, stride=1)
        torch.nn.init.xavier_uniform_(self.c3.weight)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = ReLU()
        self.fc1 = Linear(in_features=64 * 29 * 29, out_features=120)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = Linear(in_features=120, out_features=84)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = Linear(in_features=84, out_features=12)
        self.float()

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
        x = torch.flatten(x, 1)
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
        self.x_epoch = []
        self.device = device

    def batch_accuracy(self, output, target):
        # output shape: [batch, 10]
        # output = nn.functional.softmax(output, dim=1)
        # print(target.size())
        # print(target.shape[1])
        # output = output.argmax(1)
        print(output.__len__())
        acc = np.random.random((25, 21))
        for i in range(len(output)):
            diff = 0
            for j in range(len(output[i])):
                diff += math.fabs(output[i][j] - target[i][j])
                acc[i][j] = diff
            diff = diff / 21
            # acc[i] = sklearn.preprocessing.normalize(acc[i])
        acc = sklearn.preprocessing.normalize(acc)
        acc = torch.from_numpy(acc)

        # acc = torch.sum(output == target)

        return acc.cpu() * 100

    def train_step(self, dataset):
        self.model.train()
        batch_loss = []
        # batch_acc = []
        for batch in dataset:
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)

            outputs = self.model(inputs)
            # outputs = torch.tensor(outputs, dtype=torch.float64, requires_grad=True, device=device)
            # outputs = torch.clone(outputs).detach().requires_grad(True)
            outputs.to(device)
            # print("Image input ", batch[0].size())
            # print("outputs ", outputs.size())
            # print("targets ", targets.size())
            loss = self.loss(outputs, targets)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            batch_loss.append(loss.item())
            # batch_acc.append(self.batch_accuracy(outputs, targets))
        mean_loss = np.mean(batch_loss)
        print(f"Train loss: {mean_loss:.5f}")
        self.train_loss.append(mean_loss)
        # self.train_acc.append(np.mean(batch_acc))

    def validation_step(self, dataset):
        self.model.eval()
        batch_loss = []
        # batch_acc = []
        with torch.no_grad():
            for batch in dataset:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)
                batch_loss.append(loss.item())
            # batch_acc.append(self.batch_accuracy(outputs, targets))

        self.val_loss.append(np.mean(batch_loss))
        # self.val_acc.append(np.mean(batch_acc))

    def test_step(self, dataset):
        self.model.eval()
        final_outputs = []
        with torch.no_grad():
            for batch in dataset:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)

                outputs = self.model(inputs)
                predicted_outputs.append(outputs)
            # batch_acc.append(self.batch_accuracy(outputs, targets))
            iterator = outputs.shape[0]
            print("OUTPUTS LENGTH :", iterator)
            for i in range(iterator):
                tensor = outputs[i].cpu()
                array = tensor.numpy()
                for j in range(array.size - 3):
                    array[j] = ReverseNormalise(array[j], min_max_values[j, 0], min_max_values[j, 1])
                                                                                                    # ADD THE AVERAGE FLOWER BACK IN LATER!!!!!
                    array += avg_flower
                # final_outputs.append(tensor)
                save_output_json(array, i)

        # save_array = np.array(final_outputs)
        # save_array.flatten()
        # save_array.tofile("results.csv", ";")


epochs = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
learning_rate = 0.0001
batch_size = 8
lenet5 = LeNet5().to(device)
model_save_path = f"Model/training_model.model"
training_set = PhotoData(
    path_to_imgs=images_path_home,
    path_to_json=json_path_home)

predicted_outputs = []
training_set_size = len(training_set)
avg_flower = training_set.average_flower
min_max_values = training_set.parameters_min_max
train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

# validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

path_to_test_set_home = 'C:/Users/Wojciech/OneDrive/Pulpit/flowers/flower_photos/train/'
path_to_test_set_laptop = 'C:/Users/wojte/OneDrive/Pulpit/flowers/flower_photos/train'
test_set = datasets.ImageFolder(path_to_test_set_home,
                                transform=preprocess)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
n_iterations = math.ceil(training_set_size / batch_size)
model = Model(lenet5, learning_rate, device)
epoch_checkpoint = 0
# model = torch.load(model_save_path, map_location=device)
# model.eval()
for epoch in range(epochs):
    model.train_step(train_loader)
    model.x_epoch.append(epoch)
    # model.validation_step(validation_loader)
    if (epoch + 1) % 5 == 0:
        print(f'epoch {epoch + 1}/{epochs}, step{epoch + 1}/{n_iterations}, inputs {train_loader}')
    if epoch_checkpoint == 500:
        torch.save(model, model_save_path)
        epoch_checkpoint = 0
    epoch_checkpoint += 1
model.test_step(test_loader)

plt.figure(dpi=150)
plt.grid()
plt.plot(model.x_epoch, model.train_loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("Accuracy.png")
plt.show()
#
# plt.figure(dpi=150)
# plt.grid()
# plt.plot(model.train_acc)
# plt.plot(model.val_acc)
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.savefig("Accuracy.png")
