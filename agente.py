# Importamos las bibliotecas necesarias
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

# Importamos los módulos personalizados del proyecto
from ent import ent                      # Entorno del juego Subway Surfers
import progreso_n_pasos as n_step                          # Gestión de progreso por pasos
import replay_memory           # Memoria de experiencia
import red_neuronal as neural_net                     # Red neuronal
from eligibility_trace import eligibility_trace  # Trazas de elegibilidad
import prom_movimiento                       # Cálculo de promedio móvil

# (Solución para error OMP, si aparece durante la ejecución)
'''
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
'''

# Inicializamos el entorno del juego y definimos el número de acciones posibles
senv = ent()
number_actions = senv.action_space

# Creamos la IA con una red neuronal convolucional y una política softmax
cnn = neural_net.CNN(number_actions)
softmax_body = neural_net.SoftmaxBody(T=10)
ai = neural_net.AI(body=softmax_body, brain=cnn)

# Configuramos el aprendizaje por pasos n y la memoria de experiencia
n_steps = n_step.NStepProgress(ai=ai, env=senv, n_step=7)
memory = replay_memory.ReplayMemory(n_steps=n_steps, capacity=5000)

# Promedio móvil para evaluar el rendimiento del agente
ma = prom_movimiento.MovingAverage(500)

# Función para cargar el modelo previamente guardado
def load():
    if os.path.isfile('old_brain.pth'):
        print("=> Cargando modelo guardado...")
        checkpoint = torch.load('old_brain.pth')
        cnn.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Carga completa.")
    else:
        print("No se encontró ningún modelo guardado.")

# Función para guardar el estado actual del modelo
def save():
    torch.save({
        'state_dict': cnn.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, 'old_brain.pth')

# Definimos los parámetros de entrenamiento
nb_epochs = 200                        # Número de épocas de entrenamiento
optimizer = optim.Adam(cnn.parameters(), lr=0.005)  # Optimizador Adam
loss = nn.MSELoss()                    # Función de pérdida: Error Cuadrático Medio

# Descomentar si se quiere continuar desde un modelo guardado
load()

# Comenzamos el entrenamiento
for epoch in range(1, nb_epochs + 1):
    print(f"Jugando partida en época: {epoch}")
    memory.run_steps(128)  # Ejecutamos 128 pasos y guardamos en la memoria

    # Entrenamiento con muestras aleatorias de la memoria
    for batch in memory.sample_batch(64):
        inputs, targets = eligibility_trace(batch, cnn)  # Calculamos los Q-targets
        inputs, targets = Variable(inputs), Variable(targets)
        predictions, _ = cnn(inputs, None)
        loss_error = loss(predictions, targets)  # Calculamos la pérdida
        optimizer.zero_grad()        # Reiniciamos los gradientes
        loss_error.backward()        # Propagación hacia atrás
        optimizer.step()             # Actualizamos los pesos

    # Evaluamos el desempeño del modelo en esta época
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print(f"Época: {epoch}, Recompensa promedio: {avg_reward}")
    save()  # Guardamos el estado del modelo

    # Verificamos si se alcanzaron hitos importantes
    if avg_reward >= 20:
        print("¡Recompensa promedio 20 alcanzada!")
        save()
    if avg_reward >= 50:
        print("¡Recompensa promedio 50 alcanzada!")
        save()
    if avg_reward >= 100:
        print("¡Felicidades! ¡Gran rendimiento!")
        save()
        break  # Finalizamos el entrenamiento si la recompensa es suficientemente buena
