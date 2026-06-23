import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def save_plot(name, folder):
    project_dir = Path(__file__).parent
    output_dir = project_dir / folder
    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir/f'{name}.png'
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_img(img, name, folder):
    project_dir = Path(__file__).parent
    output_dir = project_dir / folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # caso especial para salvar imagens canny
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255

    img = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    img = img.astype(np.uint8)
    cv2.imwrite(str(output_dir / f'{name}.png'), img)