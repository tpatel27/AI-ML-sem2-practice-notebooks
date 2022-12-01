import cv2
import numpy as np

img = cv2.imread("../../../images/cards.jpg")


def reorder(pts):
    pts = np.array(pts).reshape((4, 2))
    pts_new = np.zeros((4, 1, 2), np.int32)
    add = pts.sum(1)
    pts_new[0] = pts[np.argmin(add)]
    pts_new[3] = pts[np.argmax(add)]
    diff = np.diff(pts, axis=1)
    pts_new[1] = pts[np.argmin(diff)]
    pts_new[2] = pts[np.argmax(diff)]
    return pts_new


w, h = 1080, 1920
pt1 = np.float32(reorder[[365, 349], [573, 338], [404, 661], [642, 643]])
pt2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
matrix = cv2.getPerspectiveTransform(pt1, pt2)
imgOutput = cv2.warpPerspective(img, matrix, (w, h))

cv2.imshow("image", img)
cv2.imshow("output", imgOutput)

cv2.waitKey(0)
