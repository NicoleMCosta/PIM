import numpy as np
import cv2

img = cv2.imread('./pimage/image1.jpeg')

(canalAzul, canalVerde, canalVermelho) = cv2.split(img)

zeros = np.zeros(img.shape[:2], dtype = "uint8")

cv2.imshow("Vermelho", canalVermelho)
cv2.imshow("Verde", canalVerde)
cv2.imshow("Azul", canalAzul)
cv2.imshow("Original", img)
cv2.waitKey(0)


# Como você interpreta o fato dos canais individuais R, G e B serem de fato matrizes (informações)
# em tons de cinza?
''''''
