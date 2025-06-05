# Identifica las esquinas superior derecha e inferior izquierda del emulador nox 
# usando imágenes predefinidas, y devuelve las coordenadas.

# Importando las librerías
import pyautogui

# La función identifica las coordenadas superior izquierda e inferior derecha del emulador nox
# Luego devuelve las coordenadas.
def begin():
    
    print("Comenzando")
    loc = {}
    loc1 = None
    loc2 = None

    while(loc1 == None):
        loc1 = pyautogui.locateOnScreen('imagenes\start_a.png', confidence=0.7) # Esperando a que aparezca la imagen de la esquina superior izquierda
 
    while(loc2 == None):
        loc2 = pyautogui.locateOnScreen('imagenes\start_b.png',confidence=0.7)  # Esperando a que aparezca la imagen de la esquina inferior derecha
    
    print("Emulador encontrado")

    loc["top"] = loc1.top + loc1.height
    loc["left"] = loc1.left
    loc["width"] = loc2.left - loc["left"]
    loc["height"] = loc2.top + loc2.height - loc["top"]

    return loc