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

#to make coordinates for the line
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

#we want to average the slope and intercept of the lines on the left and right
def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)#fits a linear equation to the points and returns a vector of coefficeints, the grad inclusive to distinguish lines on the left from right
        slope = parameters[0]
        intercept = parameters[1]
        if slope<0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line,right_line])

#import image
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image) #to avoid changing original image

#region of interest
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)

#hough transform
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40 , maxLineGap=5)#(2 & pi/180 are definining grid size)
averaged_lines = average_slope_intercept(lane_image,lines)
line_image = display_lines(lane_image,averaged_lines)

#combine line image with original image

combined_image = cv2.addWeighted(lane_image, 0.8, line_image, 1,1)#weights make on image more vivid in this case lines
#show image
cv2.imshow('result',combined_image)
cv2.waitKey(0) #waitkey 0 keeps window infinitely open till we press any key

