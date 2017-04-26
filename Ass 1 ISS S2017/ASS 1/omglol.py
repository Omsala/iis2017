# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:46:19 2017

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


#cascades from https://github.com/opencv/opencv/tree/master/data/haarcascades due to not at campus
#face_cascade  = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
#eye_cascade   = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('./cascades/haarcascade_smile.xm')

cap = cv2.VideoCapture('capture.avi')

termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

def VJFindFace(frame):   
    allRoiPts = []    
    orig = frame.copy()    
    dim = (frame.shape[1], frame.shape[0]);        
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)                
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)        
    faceRects = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (10, 10))    
    for (x, y, w, h) in faceRects:
        x = (x+10)
        y = (y+10)
        w = (w-15)
        h = (h-15)            
        allRoiPts.append((x, y, x+w, y+h))        
    cv2.imshow("Faces", frame)
    cv2.waitKey(1)  
    return allRoiPts
    
def trackFace(allRoiPts, allRoiHist):        
    for k in range(0, 30):
        ret, frame = cap.read()
        if not ret:
            return -1;
            break
        i=0
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for roiHist in allRoiHist:            
            backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
            (r, allRoiPts[i]) = cv2.CamShift(backProj, allRoiPts[i], termination)  
            for j in range(0,4):         
                if allRoiPts[i][j] < 0:
                    allRoiPts[i][j] = 0
            pts = np.int0(cv2.boxPoints(r))        
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
            i = i + 1            
        cv2.imshow("Faces", frame)
        cv2.waitKey(1)
    return 1;

def calHist(allRoiPts):
    global orig
    allRoiHist = []    
    for roiPts in allRoiPts:                        
        roi = orig[roiPts[1]:roiPts[-1], roiPts[0]:roiPts[2]]            
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)            
        roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
        roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
        allRoiHist.append(roiHist);

    return allRoiHist
        
        
def justShow():
    global cap
    for k in range(0,2):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Faces", frame)
        cv2.waitKey(1)
        
        
        
def main():
    #Include the global varibles for manipulation
    global cap
    i=0
    #While frames are present in the video
    while(cap.isOpened()):                
        #Try to find the faces using Viola-Jones. If faces are found, give the
        #pass to track it else for next five frames don't check any faces. Repeat until
        #a face is found in the frame
        if i % 2 == 0:
            #Before each call empty the pervious faces and their hsv histograms 
            allRoiPts = []
            allRoiHist = []
            
            #Read the frame and check if the frame is read. If these is some error reading the fram then return
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cv2.destroyAllWindows()
                return                
            #Capture the faces found in frame into a list                
            allRoiPts = VJFindFace(frame)
                                        
            #Check if faces are found in the given frame
            #If the face/faces are found 
            if len(allRoiPts) != 0:
                allRoiHist = calHist(allRoiPts)
                i=i+1
            else:
                #If no face is found display the next five frames without any processing 
                #To go for tracking in the next frame
                justShow()

        else:
            #Track the face found by viola jones for next TRACK number of frames using cam shift
            #print len(roiPts)
            error = trackFace(allRoiPts, allRoiHist)
            if error == -1:
                cap.release()
                cv2.destroyAllWindows()
                return
            i=i+1                

        #Exit on key press of q                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

####################################################################################################################
# call main() function

if __name__ == "__main__":
    main()