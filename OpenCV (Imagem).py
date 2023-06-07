import cv2
import numpy as np
 
input_img = cv2.imread("ntanko.png")

# Make a copy to draw contour outline
input_image_cpy = input_img.copy()
 
hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
 
# define range of red color in HSV
minimo_vermelho = np.array([0, 50, 50])
maximo_vermelho = np.array([10, 255, 255])
 
# define range of green color in HSV
minimo_verde = np.array([40, 20, 50])
maximo_verde = np.array([90, 255, 255])
 
# define range of blue color in HSV
minimo_azul = np.array([100, 50, 50])
maximo_azul = np.array([130, 255, 255])
 
# create a mask for red color
mascara_vermelha = cv2.inRange(hsv, minimo_vermelho, maximo_vermelho)
 
# create a mask for green color
mascara_verde = cv2.inRange(hsv, minimo_verde, maximo_verde)
 
# create a mask for blue color
mascara_azul = cv2.inRange(hsv, minimo_azul, maximo_azul)
 
# find contours in the red mask
contornos_vermelho, _ = cv2.findContours(mascara_vermelha, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
# find contours in the green mask
contornos_verde, _ = cv2.findContours(mascara_verde, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
# find contours in the blue mask
contornos_azul, _ = cv2.findContours(mascara_azul, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
# loop through the red contours and draw a rectangle around them
for cnt in contornos_vermelho:
    contour_area = cv2.contourArea(cnt)
    if contour_area > 1000:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(input_img, 'Vermelho', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
     
# loop through the green contours and draw a rectangle around them
for cnt in contornos_verde:
    contour_area = cv2.contourArea(cnt)
    if contour_area > 1000:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(input_img, 'Verde', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
 
# loop through the blue contours and draw a rectangle around them
for cnt in contornos_azul:
    contour_area = cv2.contourArea(cnt)
    if contour_area > 1000:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(input_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(input_img, 'Azul', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
 
# Display final output for multiple color detection opencv python
cv2.imshow('ntanko.png', input_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

