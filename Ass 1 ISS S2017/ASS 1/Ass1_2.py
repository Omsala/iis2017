# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 08:34:13 2017

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



cap = cv2.VideoCapture('capture.avi')

# Set up cascades
face_cascade  = cv2.CascadeClassifier(face_path)
eye_cascade   = cv2.CascadeClassifier(eye_path)
smile_cascade = cv2.CascadeClassifier(smile_path)

ret ,frame = cap.read()

bw = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
smile = smile_cascade.detectMultiScale(bw,1.05,1)  
eyes = eye_cascade.detectMultiScale(bw,1.05,1)
face = face_cascade.detectMultiScale(bw,1.05, 1)    

# setup initial location of window
c,r,w,h = face[0]
ce,re,we,he = eyes[0]
ce1,re1,we1,he1 = eyes[1]
cs,rs,ws,hs = smile[0]

track_window = (c,r,w,h)
track_window_eye_1 = (ce,re,we,he)
track_window_eye_2 = (ce1,re1,we1,he1)
track_window_smile = (cs,rs,ws,hs)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

#---------------------------------------------------------
roi_2 = frame[re:re+he, ce:ce+we]
hsv_roi_2 =  cv2.cvtColor(roi_2, cv2.COLOR_BGR2HSV)
mask_2 = cv2.inRange(hsv_roi_2, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist_2 = cv2.calcHist([hsv_roi_2],[0],mask_2,[180],[0,180])
cv2.normalize(roi_hist_2,roi_hist_2,0,255,cv2.NORM_MINMAX)
#---------------------------------------------------------
roi_3 = frame[re1:re1+he1, ce1:ce1+we1]
hsv_roi_3 =  cv2.cvtColor(roi_3, cv2.COLOR_BGR2HSV)
mask_3 = cv2.inRange(hsv_roi_3, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist_3 = cv2.calcHist([hsv_roi_3],[0],mask_3,[180],[0,180])
cv2.normalize(roi_hist_3,roi_hist_3,0,255,cv2.NORM_MINMAX)
#---------------------------------------------------------
roi_4 = frame[rs:rs+hs, cs:cs+ws]
hsv_roi_4 =  cv2.cvtColor(roi_4, cv2.COLOR_BGR2HSV)
mask_4 = cv2.inRange(hsv_roi_4, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist_4 = cv2.calcHist([hsv_roi_4],[0],mask_4,[180],[0,180])
cv2.normalize(roi_hist_4,roi_hist_4,0,255,cv2.NORM_MINMAX)
#---------------------------------------------------------

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10, 1)





while(1):
    ret,frame = cap.read()
    #print("-----Face-------")    
    #print (track_window)
    #print("------Smile------")
    #print(track_window_smile)
    #print("----EYES--------")
    #print(track_window_eye_1)

#---------------------------------------------------------    
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
#---------------------------------------------------------
        hsv_2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst_2 = cv2.calcBackProject([hsv_2],[0],roi_hist_2,[0,180],1)        
#---------------------------------------------------------      
        hsv_3 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst_3 = cv2.calcBackProject([hsv_3],[0],roi_hist_3,[0,180],1)        
#---------------------------------------------------------
        hsv_4 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst_4 = cv2.calcBackProject([hsv_4],[0],roi_hist_4,[0,180],1)        

#---------------------------------------------------------        
        # apply camshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        ret_eye_1, track_window_eye_1 = cv2.CamShift(dst_2, track_window_eye_1, term_crit)
        ret_eye_2, track_window_eye_2 = cv2.CamShift(dst_3, track_window_eye_2, term_crit)
        ret_smile, track_window_smile = cv2.CamShift(dst_4, track_window_smile, term_crit)
#---------------------------------------------------------
        # Draw it on image FACE BLUE
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)        
        img2 = cv2.polylines(frame,[pts],True, (255,0,0),2)
 #---------------------------------------------------------      
        # Draw it on image EYE ONE Green
        pts1 = cv2.boxPoints(ret_eye_1)
        pts1 = np.int0(pts1)
        img2 = cv2.polylines(frame,[pts1],True, (0,255,0),2)
#---------------------------------------------------------      
        # Draw it on image EYE TWO Red
        pts2 = cv2.boxPoints(ret_eye_2)
        pts2 = np.int0(pts2)
        img2 = cv2.polylines(frame,[pts2],True, (0,0,255),2)
#---------------------------------------------------------        
        # Draw it on image SMILE BLACK
        pts3 = cv2.boxPoints(ret_smile)
        pts3 = np.int0(pts3)
        #img2 = cv2.polylines(img2,[pts3],True, (0,0,0),2)
#---------------------------------------------------------        

        cv2.imshow('OutVid',img2)
        #outVideo.write(img2)
        #cv2.imshow('StartFrame',bw)
        #cv2.imshow('hsv_roi_2-- EYE',hsv_roi_2)
        #cv2.imshow('SMILE',hsv_roi_4)        
        #cv2.imshow('hsv_roi',hsv_roi)
        #cv2.imshow('MASK--EYE',mask_4)
                
      
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break   
                
    else:
        break
cap.release()
cv2.destroyAllWindows()