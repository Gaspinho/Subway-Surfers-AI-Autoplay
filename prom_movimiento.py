# Clase para calcular el promedio móvil de los últimos 'n' valores.
# Útil para monitorear el desempeño reciente de un agente de RL u otro modelo.

import numpy as np

class MovingAverage:
    def __init__(self, size):
        # Número máximo de valores a mantener para calcular el promedio
        self.size = size
        self.values = []

    def add(self, reward):
        # Agrega uno o varios nuevos valores a la lista
        if isinstance(reward, list):
            self.values += reward
        else:
            self.values.append(reward)

        # Si se excede el tamaño permitido, elimina los valores más antiguos
        while len(self.values) > self.size:
            self.values.pop(0)

    def average(self):
        # Calcula el promedio de los valores actuales
        return np.mean(self.values)