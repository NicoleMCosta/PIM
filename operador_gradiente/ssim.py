import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import cv2
from pathlib import Path
import post_treatment as post

def compare():
    current_dir = Path(__file__).resolve().parent
    folder = current_dir / 'images'

    for i in folder.iterdir():
        img = cv2.imread(str(i), 0)
        imgprewit = cv2.imread(f'{current_dir}/prewitt/{i.stem}.png',0)
        imgscharr = cv2.imread(f'{current_dir}/scharr/{i.stem}.png', 0)
        imgcanny = cv2.imread(f'{current_dir}/canny/sigma1{i.stem}.png',0)
        imgcanny3 = cv2.imread(f'{current_dir}/canny/sigma3{i.stem}.png',0)

        mse_img = mean_squared_error(img, img)
        ssim_img = ssim(img, img, data_range=img.max() - img.min())

        mse_prewitt = mean_squared_error(img, imgprewit)
        ssim_prewitt = ssim(img, imgprewit, data_range=imgprewit.max() - imgprewit.min())

        mse_scharr = mean_squared_error(img, imgscharr)
        ssim_scharr = ssim(img, imgscharr, data_range=imgscharr.max() - imgscharr.min())

        mse_canny = mean_squared_error(img, imgcanny)
        ssim_canny = ssim(img, imgcanny, data_range=imgcanny.max() - imgcanny.min() )

        mse_canny3 = mean_squared_error(img, imgcanny3)
        ssim_canny3 = ssim(img, imgcanny3, data_range=imgcanny3.max() - imgcanny3.min() )


        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(30, 10))
        ax = axes.ravel()

        ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
        ax[0].set_xlabel(f'MSE: {mse_img:.2f}, SSIM: {ssim_img:.2f}')
        ax[0].set_title('Original image')

        ax[1].imshow(imgprewit, cmap=plt.cm.gray, vmin=0, vmax=255)
        ax[1].set_xlabel(f'MSE: {mse_prewitt:.2f}, SSIM: {ssim_prewitt:.2f}')
        ax[1].set_title('Filtro de Prewitt')

        ax[2].imshow(imgscharr, cmap=plt.cm.gray, vmin=0, vmax=255)
        ax[2].set_xlabel(f'MSE: {mse_scharr:.2f}, SSIM: {ssim_scharr:.2f}')
        ax[2].set_title('Filtro de Scharr')

        ax[3].imshow(imgcanny, cmap=plt.cm.gray, vmin=0, vmax=255)
        ax[3].set_xlabel(f'MSE: {mse_canny:.2f}, SSIM: {ssim_canny:.2f}')
        ax[3].set_title('Canny sigma = 1')

        ax[4].imshow(imgcanny3, cmap=plt.cm.gray, vmin=0, vmax=255)
        ax[4].set_xlabel(f'MSE: {mse_canny3:.2f}, SSIM: {ssim_canny3:.2f}')
        ax[4].set_title('Canny sigma = 5')
        ax[5].axis('off')

        plt.tight_layout()
        plt.show()
        post.save_plot(f'compara_{i.stem}', 'comparing')