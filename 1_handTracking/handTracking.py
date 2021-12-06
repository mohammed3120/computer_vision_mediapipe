import cv2
import mediapipe as mp
import time
capture = cv2.VideoCapture(0)

mpHands_file = mp.solutions.hands
fun_hands = mpHands_file.Hands()

mpDraw_file = mp.solutions.drawing_utils

pTime = 0
cTime = 0
while(True):
    ret, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = fun_hands.process(imgRGB)
    
    landmarks_all = results.multi_hand_landmarks
    
    if landmarks_all:
        #print(len(landmarks_all)) #number of hands
        for handLms in landmarks_all:
            for idd,lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if idd == 0:
                    cv2.circle(img,(cx,cy),15,(255,10,255),cv2.FILLED)
            mpDraw_file.draw_landmarks(img, handLms, mpHands_file.HAND_CONNECTIONS)    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10, 70),cv2.FONT_HERSHEY_DUPLEX,3,(0,255,255),3)
    cv2.imshow('video original', img)
      
    if cv2.waitKey(1) == 27:
        break
  
capture.release()
cv2.destroyAllWindows()