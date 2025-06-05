# Obtiene las coordenadas de la pantalla y realiza una acción controlando el mouse automáticamente.

# Importando las librerías
import pyautogui
import time

class action():
    left=0
    top=0
    width=0
    height=0
    # Pasa las coordenadas de la pantalla a la clase action.
    def __init__(self,left,top,width,height):
        self.left=left
        self.top=top
        self.width=width
        self.height=height

    # Realiza la acción controlando el mouse dentro del área delimitada y deslizándolo a las coordenadas dadas.
    def perform(self,t):
        if(t == 0): #no hacer nada
            time.sleep(0.3)
        else:
            pyautogui.mouseDown(x=self.left+self.width/2, y=self.top+self.height/2, button='left')
            #Casos para deslizar
            if(t == 1):		#izquierda
                pyautogui.mouseUp(x=self.left+self.width/2 - self.width/5 , y=self.top+self.height/2)
            elif (t == 2):	#derecha
                pyautogui.mouseUp(x=self.left+self.width/2 + self.width/5, y=self.top+self.height/2)
            elif (t == 3):	#saltar
                pyautogui.mouseUp(x=self.left+self.width/2 , y=self.top+self.height/2 - self.height/5)
            elif (t == 4):	#agacharse
                pyautogui.mouseUp(x=self.left+self.width/2 , y=self.top+self.height/2 + self.height/5)
            pyautogui.moveTo(x=self.left+self.width/2, y=self.top+self.height/2)
            time.sleep(0.15)