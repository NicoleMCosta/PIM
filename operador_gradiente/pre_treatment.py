import numpy as np
import cv2
from pathlib import Path
import post_treatment as post

def median_blur(image):
    kernel = np.ones((3,3), np.float32)
    
    h,w = image.shape
    kernel_height, kernel_width = kernel.shape
    
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT)
    
    output = np.zeros((h,w), dtype=np.uint8)
    
    for y in range(h):
            for x in range(w):
                neighborhood = []
                
                for yk in range(kernel_height):
                    for xk in range(kernel_width):
                        pixel_val = padded_image[y + yk, x + xk]
                        neighborhood.append(pixel_val)
                
                median_value = np.median(neighborhood)
                output[y, x] = int(median_value)

    return output

def apply_blur():
    current_dir = Path(__file__).resolve().parent
    folder = current_dir / 'images'
    filter_folder = current_dir / 'filtered' 
    for img in folder.iterdir():    
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        filtered_img = median_blur(image)
        post.save_img(filtered_img,f'filtered_{img.stem}', 'filtered')
    return folder, filter_folder