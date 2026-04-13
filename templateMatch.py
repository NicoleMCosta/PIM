import csv
import cv2
import numpy as np
import os

def rgb_to_gray(image):
    #opencv reads BGR
    r = image[:, :, 2]
    g = image[:, :, 1]
    b = image[:, :, 0]

    gray = 0.3*r + 0.59*g + 0.11*b
    return gray

def csv_create(method):
    with open(f'{method}_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Min', 'Max'])

    
def execution():
    if not os.path.exists('./grayframes'):
        os.makedirs('./grayframes')

    for i in os.listdir('./frames'):
            img = cv2.imread(f'./frames/{i}')
            gray_img = rgb_to_gray(img)
            cv2.imwrite(f'./grayframes/gray_{i}', gray_img)
