import numpy as np
import cv2
 
 
###### 2. Color planes

def swap(image_original):
    
    image_final = image_original.copy()

    # Swap the red and blue channel
    image_final = cv2.cvtColor(image_final, cv2.COLOR_BGR2RGB)

    # Save jpg
    cv2.imwrite('output/o-2-a-0.jpg', image_final, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    
    
def monochromeGreen(image_original):
    
    image_final = image_original.copy()
    image_final[:,:,0] = 0
    image_final[:,:,2] = 0
    
      # Save jpg
    cv2.imwrite('output/o-2-b-0.jpg', image_final, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return image_final
    
    
def monochromeRed(image_original):
    
    image_final = image_original.copy()
    image_final[:,:,0] = 0
    image_final[:,:,1] = 0
    
    # Save jpg
    cv2.imwrite('output/o-2-c-0.jpg', image_final, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return image_final


###### 3. Replacements of pixels
def pixelReplacement(image_original, monocA, monocB):

    image_final = image_original.copy()

    heightA, widthA, _ = monocA.shape 
    middleA = monocA[(heightA//2)-50:(heightA//2)+50,(widthA//2)-50:(widthA//2)+50]

    heightB, widthB, _ = monocB.shape
    monocB[(heightB//2)-50:(heightB//2)+50,(widthB//2)-50:(widthB//2)+50] = middleA
    cv2.imwrite('output/o-3-0.jpg', monocB, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    image_final[:,:,2] = monocB[:,:,2]
    cv2.imwrite('output/o-3-1.jpg', image_final, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


###### 4. Arithmetic and geometric operations
def normalizeValues(image_original):
	
	image_final = image_original.copy()
	
	mean, stdev = cv2.meanStdDev(image_final[:,:,1])	
	image_final[:,:,1] = ((image_final[:,:,1] - mean)/stdev*10).astype(int)
	image_final = image_final + image_original
	
	cv2.imwrite('output/o-4-b-0.jpg', image_final, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def shiftToLeft(image_original):
	image_final = image_original.copy()

	r,c,_ = image_final.shape	
	m = np.float32([[1,0,-2],[0,1,0]]) #shift two pixels to left (x-axis) and zero in y-axis 
	image_final = cv2.warpAffine(image_final, m, (c,r))

	cv2.imwrite('output/o-4-c-0.jpg', image_final, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

	image_final = image_original.astype(np.int16) - image_final

	cv2.imwrite('output/o-4-c-1.jpg', image_final, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


###### 5. Noise

def noiseGreen(image_original):
    
    image_final = image_original.copy()
    
    row,col,_= image_final.shape
    mean = 10
    sigma = 20
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    image_final[:,:,1] = image_final[:,:,1] + gauss

    # Save jpg
    cv2.imwrite('output/o-5-a-0.jpg', image_final, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def noiseBlue(image_original):
    
    image_final = image_original.copy()
    
    row,col,_= image_final.shape
    mean = 10
    sigma = 20
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    image_final[:,:,0] = image_final[:,:,0] + gauss
    
    # Save jpg
    cv2.imwrite('output/o-5-b-0.jpg', image_final, [int(cv2.IMWRITE_JPEG_QUALITY), 100])



# Read the image
image_original = cv2.imread('input/i-1-0.jpg')

swap(image_original)
mg = monochromeGreen(image_original)
mr = monochromeRed(image_original)
pixelReplacement(image_original, mg.copy(), mr.copy())
normalizeValues(mg)
shiftToLeft(mg)
noiseGreen(image_original)
noiseBlue(image_original)
