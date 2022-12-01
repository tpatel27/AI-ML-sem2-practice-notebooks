import cv2

fWidth = 640
fHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, fWidth)
cap.set(4, fHeight)
cap.set(10, 1920)
while True:
    success, img = cap.read()
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
