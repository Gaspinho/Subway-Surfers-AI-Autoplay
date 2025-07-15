# Devuelve la media movil de las ultimas 100 observaciones.
# Para medir el rendimiento de nuestro modelo en los ultimos 100 pasos al final de cada epoch.
# 100 es un numero aleatorio, puede ser cualquier otro numero.

import numpy as np

class MA:
    def __init__(self, size):
        self.size = size #Size of moving average
        self.list_of_rewards = []
        

    # funcion para calcular la media de la lista de recompensas.
    def average(self):
        return np.mean(self.list_of_rewards)

    # funcion para calcular la desviacion estandar de la lista de recompensas.
    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards = self.list_of_rewards + rewards
        else:
            self.list_of_rewards.append(rewards)
            
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]