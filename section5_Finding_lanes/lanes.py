#computer vision library
import cv2
import numpy as np
import matplotlib.pyplot as plt
#import image

#grayscale
#gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

#reduce noise-blur, noise can create false edges
#blur = cv2.GaussianBlur(gray,(5,5),0)

#lane dtection using gradient
#the canny() function performs a gradient across the image and identifies sharp gradients as edges
#canny = cv2.Canny(blur, 50,150)

#canny function to grayscale, blur and detect edges
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur, 50,150)
    return canny

def region_of_interest(image):
    height = image.shape[0] #image height
    triangle = np.array([[(200,height), (1100,height), (550,250)]])#fillpoly fills with more than one polygon
    mask = np.zeros_like(image)  #blackground
    cv2.fillPoly(mask,triangle,255)
    masked_image = cv2.bitwise_and(mask,image)
    return masked_image

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

canny = canny(lane_image)
cropped_image = region_of_interest(canny)

#this ensures the priginal image is not tampered with

#show image
cv2.imshow('result',cropped_image)
cv2.waitKey(0) #waitkey 0 keeps window infinitely open till we press any key

