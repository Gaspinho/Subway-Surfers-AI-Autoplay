# Codigo encargado de controlar el mouse y realizar las acciones en el juego.
import pyautogui
import time

class action():
	left=0
	top=0
	width=0
	height=0
	# Pasa las coordenadas de la pantalla a la clase de action.
	def __init__(self,left,top,width,height):
		self.left=left
		self.top=top
		self.width=width
		self.height=height

	def perform(self,t):
		if(t == 0): #do nothing
			time.sleep(0.3)
		else:
			pyautogui.mouseDown(x=self.left+self.width/2, y=self.top+self.height/2, button='left')
			#Cases for swipe
			if(t == 1):		#left
				pyautogui.mouseUp(x=self.left+self.width/2 - self.width/5 , y=self.top+self.height/2)
			elif (t == 2):	#right
				pyautogui.mouseUp(x=self.left+self.width/2 + self.width/5, y=self.top+self.height/2)
			elif (t == 3):	#jump
				pyautogui.mouseUp(x=self.left+self.width/2 , y=self.top+self.height/2 - self.height/5)
			elif (t == 4):	#roll
				pyautogui.mouseUp(x=self.left+self.width/2 , y=self.top+self.height/2 + self.height/5)
			pyautogui.moveTo(x=self.left+self.width/2, y=self.top+self.height/2)
			time.sleep(0.15)

