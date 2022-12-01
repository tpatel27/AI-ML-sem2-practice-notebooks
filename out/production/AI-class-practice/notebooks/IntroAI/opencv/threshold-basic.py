import cv2
import numpy as np

img = cv2.imread("../../../images/2022_Miami_GP_-_Red_Bull_RB18_of_Sergio_Pérez.jpg")
kernel = np.ones((5, 5), np.uint8)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
# imgCanny = cv2.Cann

cv2.imshow("Grey Image", imgGray)
cv2.imshow("Blurr Image", imgBlur)
