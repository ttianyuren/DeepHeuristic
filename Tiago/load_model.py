import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import operator
import matplotlib.pyplot as plt
import sys

class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


ROT = [0, -1, 0, 2, 1]

scalar = StandardScaler()

model = nn.Sequential(
           nn.Conv2d(1, 10, kernel_size=(3, 5)),
           nn.MaxPool2d((1, 2)),
           nn.ReLU(),
           nn.Dropout2d(),
           nn.Conv2d(10, 20, kernel_size=(3, 3)),
           nn.MaxPool2d((2, 3)),
           nn.ReLU(),
           Flatten(),
           nn.Linear(160, 80),
           nn.ReLU(),
           nn.Linear(80, 1)
         )
model.load_state_dict(torch.load("CNN_Model_8.pk"))


def predict_one_part(img, direction):
    if direction == 2:
        img = img[24:40, 24:40]
    else:
        img = np.rot90(img, ROT[direction])[-8:, 16:48]
    img = torch.tensor(img.copy())
    y_test_pred = model(torch.reshape(img, (1, 1, 8, 32)).float())
    y_test_pred = torch.sigmoid(y_test_pred)
    return [np.round(y_test_pred.detach().item() * 100, decimals=2)]


def predict_grasp_direction(img):
    img = np.reshape(img, (-1, 1))
    X_test = scalar.fit_transform(img)
    X_test = np.reshape(X_test, (64, 64))
    result = {}
    for i in range(5):
        result[i] = predict_one_part(X_test, i)
    return result
