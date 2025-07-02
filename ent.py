# Importando librerías necesarias
import numpy as np
import pyautogui
import time
from pynput import keyboard
import random

#Importando archivos necesarios
from accion import action
from start import begin
from PreProcesamiento_imagen import preprocess_image

class ent:
    def __init__(self):
        # Número de acciones posibles
        self.action_space = 5

        # Detecta la posición y tamaño del área de juego
        self.loc = begin()

        # Realiza clic en el centro del área de juego para activarlo
        pyautogui.click(
            x=self.loc["left"] + self.loc["width"] / 2,
            y=self.loc["top"] + self.loc["height"] / 2,
            clicks=1,
            button='left'
        )

        # Crea una instancia para ejecutar acciones dentro del área de juego
        self.act = action(
            self.loc["left"],
            self.loc["top"],
            self.loc["width"],
            self.loc["height"]
        )

    # Devuelve una acción aleatoria del conjunto de acciones posibles
    def action_space_sample(self):
        return random.randint(0, 4)

    # Reinicia el juego esperando que aparezca el botón "play" y retorna el estado inicial
    def reset(self):
        while pyautogui.locateOnScreen('images\\play.png', confidence=0.7) is None:
            time.sleep(0.1)
        x, y = pyautogui.locateCenterOnScreen('images\\play.png', confidence=0.7)
        pyautogui.click(x, y)
        time.sleep(2.5)
        screenshot = pyautogui.screenshot(region=(
            self.loc["left"], self.loc["top"],
            self.loc["width"], self.loc["height"]
        ))
        state = preprocess_image(screenshot)
        return state

    # Ejecuta una acción y devuelve:
    # el nuevo estado, una recompensa, y si el juego terminó
    def step(self, action):
        self.act.perform(action)

        # Verifica si el juego ha terminado (si aparece el botón "play")
        done = pyautogui.locateOnScreen('images\\play.png', confidence=0.7) is not None

        # Captura y procesa la nueva imagen del entorno
        next_state = preprocess_image(pyautogui.screenshot(region=(
            self.loc["left"], self.loc["top"],
            self.loc["width"], self.loc["height"]
        )))

        # Asigna recompensa según si terminó o no
        reward = -10 if done else 2

        return next_state, reward, done, {}