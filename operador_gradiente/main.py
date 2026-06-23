import op_gradiente as opg
import post_treatment as post
import pre_treatment as pret
import canny_edge as ced
import ssim as ssim
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def execute(name, x, y):
    folder, filter_folder = pret.apply_blur()
    for img in folder.iterdir():
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        filtered_img = cv2.imread(f'{filter_folder}/filtered_{img.stem}.png', cv2.IMREAD_GRAYSCALE)
        
        # display results
        _, ax = plt.subplots(nrows=1, ncols=3, figsize=(30,10))
        ax = ax.flatten()

        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Imagem OG', fontsize=12)

        ax[1].imshow(filtered_img, cmap='gray')
        ax[1].set_title('Filtro de Mediana', fontsize=12)

        final_img = opg.prewitt_scharr(filtered_img, x, y)
        post.save_img(final_img, img.stem, name)

        ax[2].imshow(final_img, cmap='gray')
        ax[2].set_title(f'{name}', fontsize=12)

        post.save_plot(f'{name}_{img.stem}', 'results')

def main():
    prewitt_x = np.array([[-1, 0, 1], 
                [-1, 0, 1], 
                [-1, 0, 1]], dtype=np.float32)

    prewitt_y = np.array([[-1, -1, -1], 
                    [ 0,  0,  0], 
                    [ 1,  1,  1]], dtype=np.float32)

    scharr_x = np.array([[-3, 0, 3], 
                    [-10, 0, 10], 
                    [-3, 0, 3]], dtype=np.float32)

    scharr_y = np.array([[-3, -10, -3], 
                    [ 0,  0,  0], 
                    [ 3,  10,  3]], dtype=np.float32)

    execute('prewitt', prewitt_x, prewitt_y)
    execute('scharr', scharr_x, scharr_y)
    ced.canny_apply()
    ssim.compare()


if __name__ == "__main__":
    main()