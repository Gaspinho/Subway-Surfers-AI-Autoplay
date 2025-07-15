import time
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as pyplot
import pandas as pd
from env import env
import n_step
import replay_memory
import neural_net
from eligibility_trace import eligibility_trace
import moving_avg

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# iniciando el entorno del Subway Surfer
senv = env()
number_actions = senv.action_space

# Definiendo la red neuronal y el cuerpo de softmax       
cnn = neural_net.CNN(number_actions)
softmax_body = neural_net.SoftmaxBody(T = 10)            
ai = neural_net.AI(body = softmax_body, brain = cnn)


n_steps = n_step.NStepProgress(ai = ai, env = senv, n_step = 7)
memory = replay_memory.ReplayMemory(n_steps = n_steps, capacity = 5000)

ma = moving_avg.MA(500) 

# Funcion para guardar y cargar los checkpoints creados durante el entrenamiento.
def load():
    checkpoint_path = 'checkpoint.pth'
    if os.path.isfile(checkpoint_path):
        print("=> Cargando checkpoint... ")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        cnn.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_reward = checkpoint['best_reward']
        ma.list_of_rewards = checkpoint['moving_average']
        print(f"Checkpoint cargado! (epoch {start_epoch})")
        print(f"Mejor recompensa hasta ahora: {best_reward}")
        return start_epoch, best_reward
    else:
        print("No se encontró checkpoint - Comenzando desde cero")
        return 0, 0

def save(epoch, best_reward):
    checkpoint_path = 'checkpoint.pth'
    print(f"=> Guardando checkpoint del epoch {epoch}...")
    torch.save({
        'epoch': epoch,
        'state_dict': cnn.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_reward': best_reward,
        'moving_average': ma.list_of_rewards
    }, checkpoint_path)
    print("Checkpoint guardado!")

# Training the AI
nb_epochs = 1000 # Cantidad de epocas
optimizer = optim.Adam(cnn.parameters(), lr = 0.005) #Us
loss = nn.MSELoss() 

# Cargar checkpoint si existe
start_epoch, best_reward = load()

# Comenzar entrenamiento desde el último epoch guardado
for epoch in range(start_epoch + 1, nb_epochs + 1):
    print(f"Jugando para Epoch: {epoch}/{nb_epochs}")
    print("Imprimiendo acciones")
    memory.run_steps(128) 
    
    print("Entrando al Epoch:")
    for batch in memory.sample_batch(64): 
        inputs, targets = eligibility_trace(batch, cnn) 
        inputs, targets = Variable(inputs), Variable(targets)
        predictions, hidden = cnn(inputs, None)
        loss_error = loss(predictions, targets) 
        optimizer.zero_grad() 
        loss_error.backward() 
        optimizer.step()

   
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps) 
    avg_reward = ma.average() 
    print(f"Epoch: {epoch}/{nb_epochs}, Recompensa Promedio: {avg_reward}") 
    
    # Actualizar mejor recompensa y guardar checkpoint
    if avg_reward > best_reward:
        best_reward = avg_reward
        print(f"¡Nueva mejor recompensa! {best_reward}")
    
    # Guardar checkpoint periódicamente
    if epoch % 5 == 0:  # Guardar cada 5 epochs
        save(epoch, best_reward)
    
    # Guardar en hitos importantes
    if avg_reward >= 20:
        print("¡Alcanzado 20!")
        save(epoch, best_reward)
    if avg_reward >= 50:
        print("¡Alcanzado 50!")
        save(epoch, best_reward)
    if avg_reward >= 100:
        print("¡Felicitaciones! ¡Objetivo alcanzado!")
        save(epoch, best_reward)
        break

# Guardar al finalizar el entrenamiento
save(epoch, best_reward)
