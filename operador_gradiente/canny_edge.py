import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
import pathlib
from PIL import Image
import post_treatment as post

def canny_apply():
    #alteração para ler as imagens de teste da atividade
    folder = pathlib.Path('./operador_gradiente/images')
    for img in folder.iterdir():
        image = np.array(Image.open(img).convert('L'))

        # Compute the Canny filter for two values of sigma
        edges1 = feature.canny(image)
        edges2 = feature.canny(image, sigma=3)
        post.save_img(edges1, f'sigma1{img.stem}', 'canny')
        post.save_img(edges2, f'sigma3{img.stem}', 'canny')

        # display results
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('noisy image', fontsize=20)

        ax[1].imshow(edges1, cmap='gray')
        ax[1].set_title(r'Canny filter, $\sigma=1$', fontsize=20)

        ax[2].imshow(edges2, cmap='gray')
        ax[2].set_title(r'Canny filter, $\sigma=3$', fontsize=20)

        for a in ax:
            a.axis('off')

        fig.tight_layout()
        post.save_plot(f'canny_{img.stem}', 'results/res_canny')
