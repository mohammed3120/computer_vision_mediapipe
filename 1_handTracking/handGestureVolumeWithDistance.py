import cv2
import mediapipe as mp
import time
import handTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

detector = htm.HandsDetector(num_hands= 1)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
#volume.GetVolumeRange()
#volume.SetMasterVolumeLevel(-20.0, None)

print(volume.GetVolumeRange())    


capture = cv2.VideoCapture(0)
pTime = 0
cTime = 0

while(True):
    ret, img = capture.read()
    img = detector.findHands(img)
    lms_list = detector.findPositions(img, draw = False)
    if len(lms_list) !=0:
         x1, y1 = lms_list[4][1:]
        # print("x1=",x1," ","y1=",y1)
         x2, y2 = lms_list[8][1:]
         cx,cy = (x1+x2)//2 , (y1+y2)//2
         
         cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)
         cv2.circle(img,(x2,y2),10,(255,0,255),cv2.FILLED)
         
         cv2.line(img,(x1,y1),(x2, y2),(255, 0, 255),5)
         cv2.circle(img,(cx,cy),10,(0,0,255),cv2.FILLED)
         
         length = int(np.linalg.norm(np.array([x1,y1]) - np.array([x2,y2])))
         vol = np.interp(length, [30,300],[-65.25,0])
         
         volume.SetMasterVolumeLevel(vol, None)
         
         cv2.rectangle(img, (50,150), (85,400), (0,255,0), 3)
         cv2.rectangle(img, (40,500), (40,50), (0,255,0), cv2.FILLED)
         
         
         
         w, h,_ = img.shape
         xCenter, yCenter = w//2 , h//2
         
         print("cx = ",cx, " ","xCenter = ",xCenter)
         print("cy = ",cy, " ","yCenter = ",yCenter)
         
         dist = int(np.linalg.norm(np.array([cx,cy]) - np.array([xCenter,yCenter])))
         print("dist = ", dist)
         
        
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10, 70),cv2.FONT_HERSHEY_DUPLEX,3,(0,255,255),3)
    cv2.imshow('video original', img)
      
    if cv2.waitKey(1) == 27:
        break
  
capture.release()
cv2.destroyAllWindows()