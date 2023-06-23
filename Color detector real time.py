import cv2 as cv
import numpy as np

webcam = cv.VideoCapture(0)
while(True):
      
    # Capture the video frame by frame
    ret, frame = webcam.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV) #convert the image to HSV, is just a filter

    #set range color for green
    #define mask
    lower = np.array([25, 52, 72], np.uint8) #lower range to detect multiples green variants 25, 52, 72
    upper = np.array([102, 255, 255], np.uint8) #upper range to detect multiple green variants 102
    mask = cv.inRange(hsv, lower, upper) 

    # Morphological Transform, Dilation for each color and bitwise_and operator etween imageFrame and mask determines to detect only that particular color
    kernel = np.ones((5, 5), "uint8")

    g_mask = cv.dilate(mask, kernel)
    res_green = cv.bitwise_and(frame, frame, mask = g_mask)

    #creating countor to track green
    contours, hierarchy = cv.findContours(g_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    #detect the area of the green surface and outline the object
    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if (area > 4500):
            x, y, w, h = cv.boundingRect(contour)
            imageFrame = cv.rectangle(frame, (x,y), (x + w, y + w), (0, 255, 0), 2)

            #add a name on the bounding box
            #cv.putText(imageFrame, "Green Colour", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
  
  
    # Display the resulting frame
    cv.imshow('frame', frame)
      
    #the 'd' button is set as the quitting button
    if cv.waitKey(10) & 0xFF == ord('d'):
        webcam.release()
        cv.destroyAllWindows()
        break
