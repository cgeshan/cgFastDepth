from PIL import Image
import numpy as np
import cv2

# Open an image file
image_path = "/home/cgeshan/Desktop/CMU/F23/cgORB_SLAM2/Custom/depth/1701999970085.tiff"
img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
# Get pixel data as a list of tuples (R, G, B)

# Display the pixel values
print(img)
