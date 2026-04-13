import numpy as np
import matplotlib.pyplot as plt
import cv2
import pathlib

#o import abaixo foi uma necessidade do meu os
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

folder1= pathlib.Path('crossfading/color')
folder2= pathlib.Path('crossfading/grays')

def folder_handling(*folders):
    for f in folders:
        if not f.exists():
            f.mkdir()
    for i in f.iterdir():
        i.unlink()


folder_handling(folder1, folder2)

#I(u) = (1-u)*I1 + u*I2;

def crossfade(img1, img2, u):
    new_val = ((1-u)*img1 + u*img2).astype(np.uint8)
    return new_val

def crossfade_manual(img1, img2, u):
    altura, largura, canais = img1.shape
    nova_img = np.zeros((altura, largura, canais), dtype=np.uint8)

    for y in range(altura):
        for x in range(largura):
            for c in range(canais):
                valor1 = img1[y, x, c]
                valor2 = img2[y, x, c]
                resultado = (1 - u) * valor1 + u * valor2
                
                nova_img[y, x, c] = int(resultado)
                
    return nova_img

img1 = cv2.imread('crossfading/MorrisHolidayMetallic/5873_gray.png')
img2 = cv2.imread('crossfading/MorrisHolidayMetallic/5874_gray.png')

for u in np.arange(0, 1, 0.25):
    # crossfaded = crossfade(img1, img2, u)
    crossfaded = crossfade_manual(img1, img2, u)
    cv2.imshow('Crossfaded Image', crossfaded)
    cv2.waitKey(0)
    cv2.imwrite('crossfading/grays/' + str(u) + '.png', crossfaded)


img3 = cv2.imread('crossfading/MorrisHolidayMetallic/5873.png')
img4 = cv2.imread('crossfading/MorrisHolidayMetallic/5874.png')

for u in np.arange(0, 1, 0.1):
    # crossfaded = crossfade(img3, img4, u)
    crossfaded = crossfade_manual(img3, img4, u)
    cv2.imshow('Crossfaded Image', crossfaded)
    cv2.waitKey(0)
    cv2.imwrite('crossfading/color/' + str(u) + '.png', crossfaded)

cv2.waitKey(0)