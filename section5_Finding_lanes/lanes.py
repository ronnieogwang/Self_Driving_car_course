#import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt


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

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None: 
        for line in lines:  #each line is 2D array (x1,y1,x2,y2)
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1),(x2,y2),(255,0,0),10)#(255,0,0 - bgr)
    return line_image

def region_of_interest(image):
    height = image.shape[0] #image height
    triangle = np.array([[(200,height), (1100,height), (550,250)]])#fillpoly fills with more than one polygon
    mask = np.zeros_like(image)  #blackground
    cv2.fillPoly(mask,triangle,255)
    masked_image = cv2.bitwise_and(mask,image)
    return masked_image

#import image
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image) #to avoid changing original image

#region of interest
canny = canny(lane_image)
cropped_image = region_of_interest(canny)

#hough transform
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40 , maxLineGap=5)#(2 & pi/180 are definining grid size)
line_image = display_lines(lane_image,lines)

#combine line image with original image

combined_image = cv2.addWeighted(lane_image, 0.8, line_image, 1,1)#weights make on image more vivid in this case lines
#show image
cv2.imshow('result',combined_image)
cv2.waitKey(0) #waitkey 0 keeps window infinitely open till we press any key

