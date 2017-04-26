# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 22:17:19 2017

@author: Oskar Ahlberg
"""
import cv2
import os.path

# Training Cascaded Classifier 
#http://docs.opencv.org/3.1.0/dc/d88/tutorial_traincascade.html#gsc.tab=0
# Paths to pre-trained cascade specifications
cascades_path = "/it/sw/opencv/opencv-3.1.0/share/OpenCV/haarcascades/"

face_path = os.path.join(cascades_path, "haarcascade_frontalface_default.xml")
eye_path = os.path.join(cascades_path, "haarcascade_eye.xml")
smile_path = os.path.join(cascades_path, "haarcascade_smile.xml")
assert os.path.exists(face_path)
assert os.path.exists(eye_path)
assert os.path.exists(smile_path)

# Set up cascades
face_cascade  = cv2.CascadeClassifier(face_path)
eye_cascade   = cv2.CascadeClassifier(eye_path)
smile_cascade = cv2.CascadeClassifier(smile_path)


#cascades from https://github.com/opencv/opencv/tree/master/data/haarcascades due to not at campus
#face_cascade  = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
#eye_cascade   = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('./cascades/haarcascade_smile.xm')

cap = cv2.VideoCapture('capture.avi')


while True:
   
    ret, frame = cap.read()
    bw = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(bw, 1.3, 5)
    
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_bw = bw[y:y+h, x:x+w]
        roi_nonBW = frame[y:y+h, x:x+w]
       
        eyes = eye_cascade.detectMultiScale(roi_bw)
        for(ex,ey,ew,eh) in eyes:
           cv2.rectangle(roi_nonBW,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
           
        smile = smile_cascade.detectMultiScale(roi_bw)
        for (sx,sy,sw,sh) in smile:
           cv2.rectangle(roi_nonBW,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
    
    
    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break    

cap.release()
cv2.destroyAllWindows()
