# Creando una clase de entorno para iniciar o reiniciar el juego, escaneando la imagen play.png en la pantalla.
import numpy as np
import pyautogui
import time
from pynput import keyboard
import random
import os
from action import action
from start_game import begin
from preprocess_image import preprocess_image


class env:
    def __init__(self):
        self.action_space = 5
        self.loc = begin()
        pyautogui.click(x=self.loc["left"]+self.loc["width"]/2, y=self.loc["top"]+self.loc["height"]/2, clicks=1, button='left')
        self.act = action(self.loc["left"], self.loc["top"], self.loc["width"], self.loc["height"])     
        
    # para tomar una accion aleatoria
    def action_space_sample(self):
        return random.randint(0,4)
    
    # metodo que ayuda a verificar si el boton de play esta visible en la pantalla.
    def _is_play_button_visible(self, confidence=0.7):
        try:
            
            play_image_path = 'imagenes/play.png'
            
            # revisa si la imagen play.png existe antes de buscarla
            if not os.path.exists(play_image_path):
                print(f"Error: {play_image_path} not found!")
                return False
                
            result = pyautogui.locateOnScreen(play_image_path, confidence=confidence)
            return result is not None
        except pyautogui.ImageNotFoundException:
            return False
        except Exception as e:
            print(f"Error checkiando boton de play: {e}")
            return False
    
    # metodo que obtiene el centro del boton de play en la pantalla.
    def _get_play_button_center(self, confidence=0.7):
        try:
            play_image_path = 'imagenes/play.png'
            return pyautogui.locateCenterOnScreen(play_image_path, confidence=confidence)
        except pyautogui.ImageNotFoundException:
            return None
        except Exception as e:
            print(f"Error getting play button center: {e}")
            return None
    
    def reset(self):
        print("Esperando boton de play...")
        max_attempts = 100  
        attempts = 0
        
        while attempts < max_attempts:
            if self._is_play_button_visible(confidence=0.6):  
                center = self._get_play_button_center(confidence=0.6)
                if center:
                    x, y = center
                    pyautogui.click(x, y)
                    print("Boton Play clickeado!")
                    break
                else:
                    print("Error: No se pudo obtener el centro del boton de play.")
            
            time.sleep(0.1)
            attempts += 1
            
            if attempts % 50 == 0: 
                print(f"Esperando Boton de play... (attempt {attempts})")
        
        if attempts >= max_attempts:
            print("Warning: No se pudo encontrar el boton de play despues de varios intentos.")
            # screenshot para depurar
            screenshot = pyautogui.screenshot()
            screenshot.save('debug_screenshot.png')
            print("Debug screenshot guardada en 'debug_screenshot.png'")
        
        time.sleep(2.5)
        region = (
            int(self.loc["left"]),
            int(self.loc["top"]),
            int(self.loc["width"]),
            int(self.loc["height"])
        )
        state = preprocess_image(pyautogui.screenshot(region=region))
        return state

    # En cada epoch se revisa si el juego termino.
    # Se esperan .2 segundos despues de tomar la accion y se revisa si play.png esta en la pantalla.
    # Si play.png no esta, retorna el siguiente estado.
    # Si es game over, reward = -5 en otro caso reward = 1.
    def step(self, action):
        self.act.perform(action)
        time.sleep(0.2)  
        
        Done = self._is_play_button_visible(confidence=0.6)  
        
        next_state = preprocess_image(pyautogui.screenshot(region=(
            int(self.loc["left"]), 
            int(self.loc["top"]), 
            int(self.loc["width"]), 
            int(self.loc["height"])
        )))
        
        reward = 2
        if Done:
            reward = -10
            print("Game over detectado!")
            
        return (next_state, reward, Done, {})