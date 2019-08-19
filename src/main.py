import numpy as np
import cv2
 
 
# Read the image
image_original = cv2.imread('input/i-1-0.jpg')

# Save jpg
cv2.imwrite('output/test.jpg', image_original, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
