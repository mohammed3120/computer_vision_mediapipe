import cv2
import mediapipe as mp
import time
import  poseTrackingModule as pdm
import numpy as np
capture = cv2.VideoCapture(0)
pTime = 0
cTime = 0
detector = pdm.PoseDetector()
count = 0
dirc = 0
while(True):
    ret, img = capture.read()
    img = detector.findPose(img,draw = False)
    lms_list = detector.findPositions(img,draw = False)
    if len(lms_list) !=0:
         angle = detector.getAngle(12, 14,16)
         per = np.interp(angle,[210,310],[0,100])
         if per == 100:
             if dirc == 0:
                 count+=0.5
                 dirc = 1
         if per == 0:
             if dirc == 1:
                 count+=0.5
                 dirc = 0
    cv2.rectangle(img, (0,350), (250,720), (0,255,0), cv2.FILLED)
    cv2.putText(img,str(count),(45, 470),cv2.FONT_HERSHEY_DUPLEX,3,(0,0,255),3)

        
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10, 70),cv2.FONT_HERSHEY_DUPLEX,3,(0,255,255),3)
    cv2.imshow('video original', img)
      
    if cv2.waitKey(1) == 27:
        break
  
capture.release()
cv2.destroyAllWindows()