import numpy as np
import cv2

PATH_FACE = 'haarcascade/haarcascade_frontalface_default.xml'
#PATH_FACE = "haarcascade/haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(PATH_FACE)
webcam = cv2.VideoCapture(0)
webcam.set(3, 640)
webcam.set(4, 480)
while True:
    ret, frame = webcam.read()
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=2, minSize= (30,30) , flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()
