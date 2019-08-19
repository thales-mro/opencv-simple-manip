import numpy as np
import cv2
 
 
###### 2. Color planes

def swap(image_original):
    
    image_final = image_original.copy()

    # Swap the red and blue channel
    image_final = cv2.cvtColor(image_final, cv2.COLOR_BGR2RGB)

    # Save jpg
    cv2.imwrite('o-2-a-0.jpg', image_final, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    
    
def monochromeGreen(image_original):
    
    image_final = image_original.copy()
    image_final[:,:,0] = 0
    image_final[:,:,2] = 0
    
      # Save jpg
    cv2.imwrite('o-2-b-0.jpg', image_final, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    
    
def monochromeRed(image_original):
    
    image_final = image_original.copy()
    image_final[:,:,0] = 0
    image_final[:,:,1] = 0
    
    # Save jpg
    cv2.imwrite('o-2-c-0.jpg', image_final, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


###### 3. Replacements of pixels


###### 4. Arithmetic and geometric operations


###### 5. Noise

def noiseGreen(image_original):
    
    image_final = image_original.copy()
    
    image_green = image_original.copy()
    image_green[:,:,0] = 0
    image_green[:,:,2] = 0
    
    row,col,_= image_green.shape
    mean = 10
    sigma = 20
    gauss = np.random.normal(mean,sigma,(row,col,1))
    gauss = gauss.reshape(row,col,1)
    image_green = image_green + gauss
    
    # replace the green channel
    image_final[:,:,1] = image_green[:,:,1]

    # Save jpg
    cv2.imwrite('o-5-a-0.jpg', image_final, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def noiseBlue(image_original):
    
    image_final = image_original.copy()
    
    image_green = image_original.copy()
    image_green[:,:,1] = 0
    image_green[:,:,2] = 0
    
    row,col,_= image_green.shape
    mean = 10
    sigma = 20
    gauss = np.random.normal(mean,sigma,(row,col,1))
    gauss = gauss.reshape(row,col,1)
    image_green = image_green + gauss
    
    # replace the blue channel
    image_final[:,:,0] = image_green[:,:,0]

    # Save jpg
    cv2.imwrite('o-5-b-0.jpg', image_final, [int(cv2.IMWRITE_JPEG_QUALITY), 100])



# Read the image
image_original = cv2.imread('input/i-1-0.jpg')

swap(image_original)
monochromeGreen(image_original)
monochromeRed(image_original)
noiseGreen(image_original)
noiseBlue(image_original)
