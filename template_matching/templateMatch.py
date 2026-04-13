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
                
                # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    top_left = min_loc
                else:
                    top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)

                #criando e populando o csv de cada método
                csv_create(m,i, min_val, max_val)

                cv2.rectangle(img2,top_left, bottom_right, 255, 2)
                plt.subplot(121)
                plt.imshow(result,cmap = 'gray')
                plt.title('Matching Result')
                plt.xticks([])
                plt.yticks([])

                plt.subplot(122)
                plt.imshow(img2,cmap = 'gray')
                plt.title('Detected Point')
                plt.xticks([])
                plt.yticks([])
                plt.suptitle(m)
                
                save_path = f'{output_folder}/res_{i}'
                plt.savefig(save_path)
                plt.close('all')
                    

def execution():
    
    method_apply()

execution()