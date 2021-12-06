import cv2
import mediapipe as mp
import time
capture = cv2.VideoCapture(0)

mpPose_file = mp.solutions.pose
fun_pose = mpPose_file.Pose()

mpDraw_file = mp.solutions.drawing_utils

pTime = 0
cTime = 0
while(True):
    ret, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = fun_pose.process(imgRGB)
    
    landmarks_all = results.pose_landmarks
    
    if landmarks_all:
        for idd,lm in enumerate(landmarks_all.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            if idd == 0:
                cv2.circle(img,(cx,cy),15,(255,10,255),cv2.FILLED)
        mpDraw_file.draw_landmarks(img, landmarks_all, mpPose_file.POSE_CONNECTIONS)    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10, 70),cv2.FONT_HERSHEY_DUPLEX,3,(0,255,255),3)
    cv2.imshow('video original', img)
      
    if cv2.waitKey(1) == 27:
        break
  
capture.release()
cv2.destroyAllWindows()