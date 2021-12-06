import cv2
import mediapipe as mp
import time
import math

class PoseDetector:
    def __init__(self, mode=False, model_com=1, smooth_lms=True, enable_seg=False, smooth_seg=True, detection_con=0.5, tracking_con=0.5):
        self.mode = mode
        self.model_com = model_com
        self.smooth_lms = smooth_lms
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg
        self.detection_con = detection_con
        self.tracking_con = tracking_con
        
        self.mpPose_file = mp.solutions.pose
        self.fun_Pose = self.mpPose_file.Pose(self.mode, 
                                               self.model_com, 
                                               self.smooth_lms, 
                                               self.enable_seg, 
                                               self.smooth_seg, 
                                               self.detection_con,
                                               self.tracking_con)
        self.mpDraw_file = mp.solutions.drawing_utils

    def findPose(self,img,draw=True):
        self.img = img
        self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.results = self.fun_Pose.process(self.imgRGB)
        self.landmarks_all = self.results.pose_landmarks
        if self.landmarks_all:
            if draw:
                self.mpDraw_file.draw_landmarks(self.img, self.landmarks_all, self.mpPose_file.POSE_CONNECTIONS)
        return self.img
    def findPositions(self,img, draw=True):
        self.lms_list = []
        if self.landmarks_all:
            for idd,lm in enumerate(self.landmarks_all.landmark):
                h, w, c = self.img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lms_list.append([idd,cx,cy])
                if draw:
                    cv2.circle(self.img,(cx,cy),8,(255,10,255),cv2.FILLED)
        return self.lms_list
    def getAngle(self,p1,p2,p3, draw=True):
        #cardinates
        x1, y1 = self.lms_list[p1][1:]
        x2, y2 = self.lms_list[p2][1:]
        x3, y3 = self.lms_list[p3][1:]
        #Calculate angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle = angle + 360
        #Drawing
        if draw:
            cv2.line(self.img, (x1, y1), (x2, y2), (255, 255, 255), thickness=3)
            cv2.line(self.img, (x2, y2), (x3, y3), (255, 255, 255), thickness=3)
            cv2.circle(self.img,(x1,y1),7,(0,0,255),cv2.FILLED)
            cv2.circle(self.img,(x1,y1),15,(0,0,255),2)
            
            cv2.circle(self.img,(x2,y2),7,(0,0,255),cv2.FILLED)
            cv2.circle(self.img,(x2,y2),15,(0,0,255),2)
            
            cv2.circle(self.img,(x3,y3),7,(0,0,255),cv2.FILLED)
            cv2.circle(self.img,(x3,y3),15,(0,0,255),2)
            
            cv2.putText(self.img,str(int(angle)),(x2-40, y2-20),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
        return  angle
        
if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = PoseDetector()
    while(True):
        ret, img = capture.read()
        img = detector.findPose(img)
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