import cv2
import mediapipe as mp
import time
import handTrackingModule as htm

capture = cv2.VideoCapture(0)
pTime = 0
cTime = 0
detector = htm.HandsDetector(num_hands=4)
while(True):
    ret, img = capture.read()
    img = detector.findHands(img)
    lms_list = detector.findPositions(img)
    if len(lms_list) !=0:
         print(lms_list[4])

        
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10, 70),cv2.FONT_HERSHEY_DUPLEX,3,(0,255,255),3)
    cv2.imshow('video original', img)
      
    if cv2.waitKey(1) == 27:
        break
  
capture.release()
cv2.destroyAllWindows()