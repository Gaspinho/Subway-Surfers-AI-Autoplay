# Importamos las librerías necesarias
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Implementación de trazas de elegibilidad (eligibility traces)
def eligibility_trace(batch, cnn):
    targets = []  # Aquí almacenaremos los valores objetivo (Q-targets)
    inputs = []   # Aquí guardaremos los estados iniciales de cada secuencia
    gamma = 0.99  # Factor de descuento: reduce el peso de las recompensas futuras

    # Recorremos cada secuencia (serie de pasos) en el lote (batch)
    for series in batch:
        # Tomamos el estado inicial y el final de la secuencia
        input = Variable(torch.from_numpy(np.array(
            [series[0].state, series[-1].state], dtype=np.float32)))

        # Hacemos una predicción (paso hacia adelante) con la red neuronal
        output, hidden = cnn(input)

        # Si el episodio terminó, el valor futuro es 0; si no, usamos el mejor valor estimado del estado final
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()

        # Recorremos la secuencia en orden inverso (desde el penúltimo hacia el primero)
        for step in reversed(series[:-1]):
            # Calculamos la recompensa acumulada con descuento
            cumul_reward = step.reward + gamma * cumul_reward

        # Tomamos la salida de la red para el estado inicial
        target = output[0].data

        # Actualizamos el valor estimado para la acción tomada con la recompensa calculada
        target[series[0].action] = cumul_reward

        # Guardamos el estado inicial y su nuevo valor objetivo
        state = series[0].state
        targets.append(target)
        inputs.append(state)

    return torch.FloatTensor(inputs), torch.FloatTensor(targets)
