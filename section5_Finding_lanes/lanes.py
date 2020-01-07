#computer vision library
import cv2

#import image
image = cv2.imread('test_image.jpg')
cv2.imshow('result', image)  #reslut is the window name to display image
cv2.waitKey(0) #waitkey 0 keeps window infinitely open till we press any key