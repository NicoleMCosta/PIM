import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def csv_create(method,i, min_val, max_val):
    with open(f'{method}_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i,min_val, max_val])

def method_apply():
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    template = cv2.imread('./frames/template.png', 0)
    w, h = template.shape[::-1]

    for m in methods:
        #criando pasta para resultados de cada método
        output_folder = f'./results_{m}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for i in os.listdir('./frames'):
            img = cv2.imread(f'./frames/{i}', 0)

            if i != 'template.png':
                img2 = img.copy()
                method = eval(m)

                result = cv2.matchTemplate(img2, template, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    top_left = min_loc
                else:
                    top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)

                #criando e populando o csv de cada método
                csv_create(m,i, min_val, max_val)

                #plotando imagem resultado e salvando
                cv2.rectangle(img2,top_left, bottom_right, 255, 2)
                save_path = f'{output_folder}/res_{i}.jpg'
                cv2.imwrite(save_path, img2)
                cv2.destroyAllWindows()
            

method_apply()
