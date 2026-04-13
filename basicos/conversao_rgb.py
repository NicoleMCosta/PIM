import numpy as np
import matplotlib.pyplot as plt

image = plt.imread('basicos/pimage/image1.jpeg')


def rgb_to_gray(image):
    #pyplot reads RGB but opencv reads BGR
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    gray = 0.3*r + 0.59*g + 0.11*b
    return gray

grayscale = rgb_to_gray(image)
plt.axis('off')
plt.imshow(grayscale, cmap='gray')
plt.savefig('basicos/pimage/gray_image1.png')
plt.waitforbuttonpress(0)