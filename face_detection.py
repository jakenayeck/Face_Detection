import numpy as np
import cv2
import sys

def initialize_camera():
    webcam = cv2.VideoCapture(0)
    webcam.set(3, 640)
    webcam.set(4, 480)
    return webcam

def get_frames(webcam):
    ret, frame = webcam.read()
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray

def draw_rectangle_over_faces(img, faces, text):
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0))
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        sys.exit()

def create_face_learner():
    PATH_FACE = 'haarcascade/haarcascade_frontalface_default.xml'
    face_learner = cv2.CascadeClassifier(PATH_FACE)
    return face_learner

def find_faces(gray_image):
    alpha = 1.2
    faces = face_learner.detectMultiScale(gray_image, scaleFactor = alpha, minNeighbors=2, minSize= (30,30) )
    return faces

################
## Main Loop  ##
################

webcam = initialize_camera()
face_learner = create_face_learner()
while True:
    color_image, gray_image = get_frames(webcam)
    faces = find_faces(gray_image)
    draw_rectangle_over_faces(color_image, faces, text="found human")

