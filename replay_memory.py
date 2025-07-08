# Implementación de "Experience Replay" usando la clase ReplayMemory
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque  # Estructura de cola doblemente terminada (útil para eliminar del inicio)

# Clase que almacena las experiencias del agente: estado, acción, recompensa y siguiente estado.
class ReplayMemory:

    # Constructor de la clase. Recibe el objeto n_steps y la capacidad máxima del buffer.
    def __init__(self, n_steps, capacity=1000):
        self.buffer = deque()               # Creamos una lista tipo cola (buffer) vacía
        self.capacity = capacity            # Máximo número de experiencias a guardar
        self.n_steps_iter = iter(n_steps)   # Obtenemos el iterador del generador de secuencias n-step
        self.n_steps = n_steps              # Guardamos el objeto n_steps por si se necesita más adelante

    # Función que ejecuta el entorno varias veces para recolectar 'samples' experiencias
    def run_steps(self, samples): 
        while samples > 0:
            samples -= 1
            entry = next(self.n_steps_iter)  # Ejecuta el entorno y obtiene una secuencia n-step
            self.buffer.append(entry)        # Guarda esa experiencia en el buffer

        # Si el buffer supera su capacidad, se eliminan las experiencias más antiguas
        while len(self.buffer) > self.capacity:
            self.buffer.popleft()

    # Generador que produce lotes aleatorios de experiencias para entrenamiento
    def sample_batch(self, batch_size):
        vals = list(self.buffer)             # Convertimos el buffer en una lista
        np.random.shuffle(vals)              # Mezclamos aleatoriamente las experiencias

        offset = 0
        while (offset + 1) * batch_size <= len(self.buffer):
            # Devuelve un lote de tamaño 'batch_size'
            yield vals[offset * batch_size : (offset + 1) * batch_size]
            offset += 1
