import cv2
import mediapipe as mp
import time

class HandsDetector:
    def __init__(self, mode=False, num_hands=2,model_com=1, detection_con=0.5, tracking_con=0.5):
        self.mode = mode
        self.num_hands = num_hands
        self.model_com = model_com
        self.detection_con = detection_con
        self.tracking_con = tracking_con
        self.mpHands_file = mp.solutions.hands
        self.fun_hands = self.mpHands_file.Hands(self.mode, self.num_hands, self.model_com, self.detection_con, self.tracking_con)
        self.mpDraw_file = mp.solutions.drawing_utils
        self.fingerTips = [4,8,12,16,20]
        

    def findHands(self,img,draw=True):
        self.img = img
        self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.results = self.fun_hands.process(self.imgRGB)
        self.landmarks_all = self.results.multi_hand_landmarks
        if self.landmarks_all:
            #print(len(self.landmarks_all)) #number of hands
            for handLms in self.landmarks_all:
                if draw:
                    self.mpDraw_file.draw_landmarks(self.img, handLms, self.mpHands_file.HAND_CONNECTIONS)
        return self.img
    def findPositions(self,img,handNo=0,draw=True):
        self.lms_list = []
        if self.landmarks_all:
            #print(len(self.landmarks_all)) #number of hands
            for handLm in self.landmarks_all:
                for idd,lm in enumerate(handLm.landmark):
                            h, w, c = self.img.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            self.lms_list.append([idd,cx,cy])
                            if draw:
                                cv2.circle(self.img,(cx,cy),8,(255,10,255),cv2.FILLED)
        return self.lms_list
    def fingersUp(self,ltr = True):
        fingers = []
        #Thumbs
        if ltr:
            if self.lms_list[4][1] > self.lms_list[6][1]:
             fingers.append(1)
            else:
                 fingers.append(0)
        else:
            if self.lms_list[4][1] < self.lms_list[6][1]:
             fingers.append(1)
            else:
                 fingers.append(0)
         #Other fingers
        for i in range(1,len(self.fingerTips)):
             if self.lms_list[self.fingerTips[i]][2] < self.lms_list[self.fingerTips[i]-2][2]:
                 fingers.append(1)
             else:
                 fingers.append(0)
        return fingers
if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = HandsDetector(num_hands=4)
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