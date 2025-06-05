# La función recibe una imagen y devuelve una versión en escala de grises y baja resolución de la imagen.

# Importando las librerías
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage.transform import resize

# Función para convertir la imagen a escala de grises y redimensionarla a 128 x 128, luego retorna esta imagen preprocesada.
def preprocess_image(img):
    img_size = (128,128,3)
    img = resize(np.array(img), img_size)
    img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114]) #Fuente - https://es.wikipedia.org/wiki/Escala_de_grises#Codificaci%C3%B3n_de_luma_en_sistemas_de_video
    img_gray = resize(img_gray, (128, 128))
    return np.expand_dims(img_gray, axis=0)





# imagen = img.imread('imagenes/temp1.png')
# s_g = preprocess_image(imagen)
# plt.figure(figsize=(12,8))
# plt.imshow(s_g, cmap=plt.get_cmap('gray'))
# plt.axis('off')
# plt.show()