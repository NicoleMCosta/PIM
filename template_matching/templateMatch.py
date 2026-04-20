import csv
import cv2
import os
import pandas as pd

def csv_create(method,i, min_val, max_val):
    with open(f'{method}_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i,min_val, max_val])

def csv_analysis(method,i, true_loc, found_loc, is_fp):
    if not os.path.exists('./analysis'):
        os.makedirs('./analysis')

    with open(f'./analysis/{method}_analysis.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i, true_loc, found_loc, is_fp])

def check_false_positive(next_val, last_val, tolerance, method, frame):
    distancia = abs(next_val - last_val)
    is_fp = distancia > tolerance
    csv_analysis(method, frame, last_val, next_val, is_fp)
    return is_fp

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
                    score_atual = min_val
                else:
                    top_left = max_loc
                    score_atual = max_val
                bottom_right = (top_left[0] + w, top_left[1] + h)

                #lê a última localização no csv para a comparação com o score atual
                #para o primeiro frame, o csv não existirá, então o código ignora
                if os.path.exists(f'./{m}_results.csv'):
                    df = pd.read_csv(f'./{m}_results.csv', header=None)
                    if not df.empty:
                        last_line = df.tail(1)
                        if 'SQDIFF' in m:
                            last_score = last_line.iloc[0, 1]#min_val
                        else:
                            last_score = last_line.iloc[0, 2]#max_val
                        
                        #adaptado para a normalizada, já que variam apenas entre 0 e 1
                        if 'NORMED' in m:
                                tolerance = 0.2
                        else:
                            tolerance = last_score * 0.15
                        is_fp = check_false_positive(score_atual, last_score, tolerance, m, i)
                
                #criando e populando o csv de cada método
                csv_create(m,i, min_val, max_val)

                #plotando imagem resultado e salvando
                cv2.rectangle(img2,top_left, bottom_right, 255, 2)
                save_path = f'{output_folder}/res_{i}.jpg'
                cv2.imwrite(save_path, img2)

              
                cv2.destroyAllWindows()
            

method_apply()
