# Las definiciones y declaraciones de la estructura de la red neuronal y la función softmax.
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as pyplot
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Creando la mente
class CNN(nn.Module):
    # Definiendo la estructura de la red neuronal.
    # 3 capas convolucionales -> 1 capa LSTM -> 2 capas lineales
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5) 
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3) 
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2) 
        self.out_neurons = self.count_neurons((1, 128, 128)) 
        self.lstm = nn.LSTMCell(self.count_neurons((1, 128, 128)), 256) 
        self.fc1 = nn.Linear(in_features = 256, out_features = 40) 
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions) 

    # Funcion para la propagación hacia adelante de los datos en la red neuronal.
    def forward(self, x, hidden=None):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2)) 
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2)) 
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2)) 
        x = x.view(-1, self.out_neurons) 
        hx, cx = self.lstm(x, hidden) 
        x = hx 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return x, (hx, cx)

    # Funcion para contar la cantidad de neuronas en la capa LSTM. 
    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2)) 
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2)) 
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2)) 
        return x.data.view(1, -1).size(1)

# Declara el modelo AI, que realiza la propagación hacia adelante utilizando la red neuronal y softmax.
class AI:

    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs, hidden):
        output, (hx, cx) = self.brain(inputs,hidden)
        actions = self.body(output)
        return actions.data.cpu().numpy(), (hx, cx)

# Cuerpo de la funcion softmax.
class SoftmaxBody(nn.Module):
    
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T 

    def forward(self, outputs):
        probs = F.softmax(outputs * self.T,dim = 0) 
        actions = probs.multinomial(num_samples=1)  
        return actions