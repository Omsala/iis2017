# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 13:04:27 2017

@author: osah3299
"""

import cv2
import os.path
import numpy as np

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
frames = 10

#cascades from https://github.com/opencv/opencv/tree/master/data/haarcascades due to not at campus
#face_cascade  = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
#eye_cascade   = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('./cascades/haarcascade_smile.xm')

def camshift(img1, img2, bb):
        hsv = cv2.cvtColor(img1, cv2.CV_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        x0, y0, w, h = bb
        x1 = x0 + w -1
        y1 = y0 + h -1
        hsv_roi = hsv[y0:y1, x0:x1]
        mask_roi = mask[y0:y1, x0:x1]
        hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
        hist_flat = hist.reshape(-1)
        prob = cv2.calcBackProject([hsv,cv2.cvtColor(img2, cv2.CV_BGR2HSV)], [0], hist_flat, [0, 180], 1)
        prob &= mask
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        new_ellipse, track_window = cv2.CamShift(prob, bb, term_crit)
        return track_window
 
def faceTracking():
    cap = cv2.VideoCapture('capture.avi')
    img = cap.read()
    bb = (125,125,200,100) # get bounding box from some method
    while True:
        try:
            img1 = cap.read()
            bb = camshift(img1, img, bb)
            img = img1
            #draw bounding box on img1
            cv2.imshow("CAMShift",img1)
        except KeyboardInterrupt:
            break