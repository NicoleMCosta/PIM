import op_gradiente as opg
import post_treatment as post
import pre_treatment as pret
import canny_edge as ced
import ssim as ssim
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


current_dir = Path(__file__).resolve().parent

def execute(name, x, y):
    folder = current_dir / 'images'

    for img in folder.iterdir():
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        filtered_img = pret.median_blur(image)

        final_img = opg.prewitt_scharr(filtered_img, x, y)
        post.save_img(final_img, img.stem, name)
        
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Original', fontsize=20)

        ax[1].imshow(filtered_img, cmap='gray')
        ax[1].set_title('Filtrada', fontsize=20)

        ax[2].imshow(final_img, cmap='gray')
        ax[2].set_title(f'{name}', fontsize=20)
        for a in ax:
            a.axis('off')
        fig.tight_layout()
        # plt.show()
        post.save_plot(f'{name}_{img.stem}', 'results')
        plt.close()

def main():
    prewitt_x = np.array([[-1, 0, 1], 
                        [-1, 0, 1], 
                        [-1, 0, 1]], dtype=np.float32)

    prewitt_y = np.array([[1, 1, 1], 
                        [ 0,  0,  0], 
                        [ -1,  -1,  -1]], dtype=np.float32)

    scharr_x = np.array([[-3, 0, 3], 
                        [-10, 0, 10], 
                        [-3, 0, 3]], dtype=np.float32)

    scharr_y = np.array([[+3, +10, +3], 
                        [ 0,  0,  0], 
                        [ -3,  -10,  -3]], dtype=np.float32)

    execute('prewitt', prewitt_x, prewitt_y)
    execute('scharr', scharr_x, scharr_y)
    ced.canny_apply()
    ssim.compare()


if __name__ == "__main__":
    main()