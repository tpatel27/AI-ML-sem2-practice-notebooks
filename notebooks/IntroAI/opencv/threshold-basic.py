import cv2
import numpy as np

img = cv2.imread("../../../images/iris_setosa.jpg")
gif = cv2.imread("../../../images/leonardo-dicaprio.gif")
kernel = np.ones((5, 5), np.uint8)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlurGaussian = cv2.GaussianBlur(imgGray, (7, 7), cv2.BORDER_CONSTANT)
imgBlurMedian = cv2.medianBlur(img, 5)
imgBlurBilateral = cv2.bilateralFilter(img,9,75,75)
imgCanny = cv2.Canny(img, 150, 200)
imgDialated = cv2.dilate(imgCanny, kernel, iterations=1)
gifCanny = cv2.Canny(gif, 150, 200)
imgEroded = cv2.erode(imgDialated, kernel, iterations=1)

# cv2.imshow("Actual Image", img)
# cv2.imshow("Grey Image", imgGray)
# cv2.imshow("Gaussian Image", imgBlurGaussian)
# cv2.imshow("Median Image", imgBlurMedian)
# cv2.imshow("Bilateral Image", imgBlurBilateral)
cv2.imshow("Canny Image", imgCanny)
cv2.imshow("Dialation Image", imgDialated)
cv2.imshow("Erode Image", imgEroded)
# cv2.imshow("GIF test", gifCanny)
cv2.waitKey(0)
cv2.destroyAllWindows()
