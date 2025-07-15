import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Implementando Eligibility Trace
def eligibility_trace(batch, cnn):
    targets = [] #el targets de la red neuronal
    inputs = []
    gamma = 0.99 #Gamma para reducir el efecto de las recompensas antiguas
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        output, hidden = cnn(input) #Forward propagation
        cumul_reward = 0.0 if series[-1].done else output[1].data.max() # definiendo la recompensa acumulada
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward # reduciendo el efecto de las recompensas antiguas
        target = output[0].data
        target[series[0].action] = cumul_reward
        state = series[0].state
        targets.append(target)
        inputs.append(state)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)