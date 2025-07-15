# Se identifica las esquinas superior derecha e inferior izquierda del emulador nox
# utilizando imágenes predefinidas y se devuelven las coordenadas.
import pyautogui

def begin():
	print("Comenzando")
	loc = {}
	loc1 = None
	loc2 = None
	while(loc1 == None):
		loc1 = pyautogui.locateOnScreen('imagenes\start_t.png', confidence=0.7)
	while(loc2 == None):
		loc2 = pyautogui.locateOnScreen('imagenes\start_b.png',confidence=0.7)
	loc["top"] = loc1.top + loc1.height
	loc["left"] = loc1.left
	loc["width"] = loc2.left - loc["left"]
	loc["height"] = loc2.top + loc2.height - loc["top"]
	return loc