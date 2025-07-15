# Implementing Experience Replay using ReplayMemory class.
# Implementando la Experience Replay utilizando la clase ReplayMemory.  
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

# Guardando la tupla del estado anterior, acción, recompensa y siguiente estado, es decir, las experiencias del agente y almacenándolas en un buffer.
class ReplayMemory:

    
    def __init__(self, n_steps, capacity = 1000):
        self.buffer = deque() 
        self.capacity = capacity 
        self.n_steps_iter = iter(n_steps) 
        self.n_steps = n_steps 

    def run_steps(self, samples): 
        while samples > 0:
            samples -= 1
            entry = next(self.n_steps_iter) 
            self.buffer.append(entry) 
        while len(self.buffer) > self.capacity: 
            self.buffer.popleft()

   
    def sample_batch(self, batch_size): 
        vals = list(self.buffer)
        np.random.shuffle(vals)
        offset = 0
        while (offset+1)*batch_size <= len(self.buffer):
            yield vals[offset*batch_size:(offset+1)*batch_size]
            offset += 1


