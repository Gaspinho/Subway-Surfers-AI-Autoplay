# Recoge secuencias de experiencias del agente en el entorno de juego,
# entregando tuplas de n pasos para el entrenamiento por refuerzo.

import torch
from torch.autograd import Variable
import numpy as np
from collections import namedtuple, deque

# Representación de un paso del agente
Step = namedtuple('Step', ['state', 'action', 'reward', 'done', 'lstm'])

class NStepProgress:
    def __init__(self, env, ai, n_step):
        self.env = env              # Entorno del juego
        self.ai = ai                # Agente con red neuronal y política
        self.n_step = n_step        # Largo de las secuencias a recolectar
        self.rewards = []           # Recompensas por episodio

    def __iter__(self):
        state = self.env.reset()
        history = deque()           # Buffer para los pasos recientes
        reward = 0.0
        is_done = True
        end_buffer = []             # Evita capturar pasos inválidos al final del juego

        while True:
            if is_done:
                cx = Variable(torch.zeros(1, 256))
                hx = Variable(torch.zeros(1, 256))
            else:
                cx = Variable(cx.data)
                hx = Variable(hx.data)

            # Obtener acción del agente
            action, (hx, cx) = self.ai(
                Variable(torch.from_numpy(np.array([state], dtype=np.float32))),
                (hx, cx)
            )

            # Añadir al buffer de finalización
            end_buffer.append((state, action))
            if len(end_buffer) > 3:
                del end_buffer[0]

            # Mostrar acción por consola
            a = action[0][0]
            actions_map = {0: "do nothing", 1: "left", 2: "right", 3: "jump", 4: "roll"}
            print(actions_map.get(a, "unknown"))

            # Ejecutar acción en el entorno
            next_state, r, is_done, _ = self.env.step(action)

            # Ajuste si termina el juego
            if is_done:
                print("\nGame Ended\n")
                if len(end_buffer) >= 3:
                    state, action = end_buffer[-3]
                    history.pop()
                r = -10

            reward += r

            # Guardar paso actual
            history.append(Step(
                state=state,
                action=action,
                reward=r,
                done=is_done,
                lstm=(hx, cx)
            ))

            # Mantener historia dentro de tamaño permitido
            while len(history) > self.n_step + 1:
                history.popleft()

            # Si hay una secuencia completa, la entregamos
            if len(history) == self.n_step + 1:
                yield tuple(history)

            state = next_state

            # Si terminó el juego, entregar lo que queda en buffer
            if is_done:
                if len(history) > self.n_step + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()

                # Reiniciar juego
                self.rewards.append(reward)
                reward = 0.0
                state = self.env.reset()
                end_buffer.clear()
                history.clear()

    def rewards_steps(self):
        # Retorna las recompensas acumuladas por episodio
        collected_rewards = self.rewards
        self.rewards = []
        return collected_rewards 