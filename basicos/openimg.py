import cv2
import numpy as np

# Reading the image using imread() function
image = cv2.imread('./pimage/image1.jpeg')

# Displaying the original BGR image
cv2.imshow('Image', image)
# print (np.mean(image)) #132.39873290389093

def mean_nonp(image):
    total = 0
    px_count = 0
    for a in image:
        for pixel in a:
            for value_rgb in pixel:
                total += int(value_rgb)
                px_count+=1

    return total/px_count

print(mean_nonp(image))

# Waits for user to press any key
cv2.waitKey(0)
