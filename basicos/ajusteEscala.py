import numpy as np
import cv2

def rgb_to_gray(image):
    #opencv is BGR
    r = image[:, :, 2]
    g = image[:, :, 1]
    b = image[:, :, 0]

    gray = 0.3*r + 0.59*g + 0.11*b
    return gray

def offscale(img):
    off_img = img + 400
    return off_img

def get_min_max(img):
    min_val = img[0][0]
    max_val = img[0][0]
    for row in img:
        for pixel in row:
            if pixel < min_val:
                min_val = pixel
            if pixel > max_val:
                max_val = pixel
    return min_val, max_val

def adjust_scale(image, min_val, max_val):
    adjusted = ((image - min_val)/ max_val)*255
    return adjusted.astype(np.uint8)

def adjust_manual(img, min, max):
    new_img = np.zeros(img.shape, dtype=np.uint8)
    for altura in range(img.shape[0]):
        for largura in range(img.shape[1]):
            new_img[altura][largura] = ((img[altura][largura] - min) / max) * 255
    return new_img


img = cv2.imread('./pimage/image1.jpeg')

image_gray = rgb_to_gray(img)
off_image = offscale(image_gray)
min_val, max_val = get_min_max(off_image)

# adjusted_image = adjust_scale(off_image, min_val, max_val)
adjusted_image = adjust_manual(off_image, min_val, max_val)
cv2.imshow('Adjusted Image', adjusted_image)
cv2.waitKey(0)
