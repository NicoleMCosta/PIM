import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

import cv2
from pathlib import Path
import post_treatment as post

current_dir = Path(__file__).resolve().parent

def compare():
    folder = current_dir / 'images'

    for i in folder.iterdir():
        imgprewit = cv2.imread(f'{current_dir}/prewitt/{i.stem}.png',0)
        imgscharr = cv2.imread(f'{current_dir}/scharr/{i.stem}.png', 0)
        imgcanny = cv2.imread(f'{current_dir}/canny/sigma1{i.stem}.png',0)
        imgcanny3 = cv2.imread(f'{current_dir}/canny/sigma3{i.stem}.png',0)
        
        dr_prewitt = float(imgprewit.max() - imgprewit.min())
        dr_scharr = float(imgscharr.max() - imgscharr.min())
        
        score_pc1, grad_pc1 = ssim(imgprewit, imgcanny, data_range=dr_prewitt, full=True)
        score_pc3, grad_pc3 = ssim(imgprewit, imgcanny3, data_range=dr_prewitt, full=True)
        score_sc1, grad_sc1 = ssim(imgscharr, imgcanny, data_range=dr_scharr, full=True)
        score_sc3, grad_sc3 = ssim(imgscharr, imgcanny3, data_range=dr_scharr, full=True)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
        ax = axes.ravel()

        ax[0].imshow(grad_pc1, cmap=plt.cm.gray, vmin=0, vmax=1)
        ax[0].set_xlabel(f'SSIM: {score_pc1:.4f}')
        ax[0].set_title(r'Prewitt x Canny $\sigma=1$')

        ax[1].imshow(grad_pc3, cmap=plt.cm.gray, vmin=0, vmax=1)
        ax[1].set_xlabel(f'SSIM: {score_pc3:.4f}')
        ax[1].set_title(r'Prewitt x Canny $\sigma=3$')

        ax[2].imshow(grad_sc1, cmap=plt.cm.gray, vmin=0, vmax=1)
        ax[2].set_xlabel(f'SSIM: {score_sc1:.4f}')
        ax[2].set_title(r'Scharr x Canny $\sigma=1$')

        ax[3].imshow(grad_sc3, cmap=plt.cm.gray, vmin=0, vmax=1)
        ax[3].set_xlabel(f'SSIM: {score_sc3:.4f}')
        ax[3].set_title(r'Scharr x Canny $\sigma=3$')

        plt.tight_layout(h_pad=0.3)
        post.save_plot(f'compara_{i.stem}', 'comparing')
        plt.close() 
        
    print('\n\nImagens resultado podem ser acessadas na pasta COMPARING\n\n')
        