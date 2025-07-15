# Funcon que toma una imagen y devuelve la versión en escala de grises de baja resolución de la imagen.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage.transform import resize

# Funcion para hacer la imagen en escala de grises y redimensionarla a 128 X 128, luego devolver esta imagen preprocesada.
def preprocess_image(img):
	img_size = (128,128,3)
	img = resize(np.array(img), img_size)
	img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114]) #referencia- https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems
	img_gray = resize(img_gray, (128, 128))
	return np.expand_dims(img_gray, axis=0)
