# Define la arquitectura de la red neuronal (CNN + LSTM) y la política basada en softmax.
#librerias
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Red neuronal principal con capas convolucionales, una LSTM y capas lineales.
class CNN(nn.Module):
    def __init__(self, number_actions):
        super(CNN, self).__init__()

        # Capas convolucionales para extracción de características
        self.convolution1 = nn.Conv2d(1, 32, kernel_size=5)
        self.convolution2 = nn.Conv2d(32, 32, kernel_size=3)
        self.convolution3 = nn.Conv2d(32, 64, kernel_size=2)

        # Se calcula automáticamente el número de neuronas tras convoluciones
        self.out_neurons = self.count_neurons((1, 128, 128))

        # LSTM para mantener una memoria del estado anterior
        self.lstm = nn.LSTMCell(self.out_neurons, 256)

        # Capas lineales para generar la acción final
        self.fc1 = nn.Linear(256, 40)
        self.fc2 = nn.Linear(40, number_actions)

    def forward(self, x, hidden=None):
        # Paso hacia adelante en la red
        x = F.relu(F.max_pool2d(self.convolution1(x), kernel_size=3, stride=2))
        x = F.relu(F.max_pool2d(self.convolution2(x), kernel_size=3, stride=2))
        x = F.relu(F.max_pool2d(self.convolution3(x), kernel_size=3, stride=2))

        # Aplanamiento del resultado
        x = x.view(-1, self.out_neurons)

        # Paso por la LSTM
        hx, cx = self.lstm(x, hidden)

        # Paso por capas completamente conectadas
        x = F.relu(self.fc1(hx))
        x = self.fc2(x)

        return x, (hx, cx)

    def count_neurons(self, image_dim):
        # Simula una pasada por la red para calcular el tamaño final
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        return x.data.view(1, -1).size(1)

# Encapsula la red neuronal y la política de decisión
class AI:
    def __init__(self, brain, body):
        self.brain = brain  # Red neuronal
        self.body = body    # Política de softmax

    def __call__(self, inputs, hidden):
        # Ejecuta el modelo y retorna la acción elegida
        output, (hx, cx) = self.brain(inputs, hidden)
        actions = self.body(output)
        return actions.data.cpu().numpy(), (hx, cx)

# Política basada en softmax
class SoftmaxBody(nn.Module):
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T  # Temperatura para controlar exploración

    def forward(self, outputs):
        probs = F.softmax(outputs * self.T, dim=0)
        actions = probs.multinomial(num_samples=1)
        return actions