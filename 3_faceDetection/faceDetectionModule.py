import cv2
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, detection_con=0.5, model_selection= 0):
        self.detection_con = detection_con
        self.model_selection = model_selection
        
        self.mpFaceDetection_file = mp.solutions.face_detection
        self.fun_FaceDetection = self.mpFaceDetection_file.FaceDetection(self.detection_con,self.model_selection)
        self.mpDraw_file = mp.solutions.drawing_utils

    def findFaces(self,img, draw = True):
        self.img = img
        self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.results = self.fun_FaceDetection.process(self.imgRGB)
        self.detections_all = self.results.detections
        bboxs = []
        if self.detections_all:
            for idd,detection in enumerate(self.detections_all):
                bboxc = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih),\
                    int(bboxc.width * iw), int(bboxc.height * ih)
                bboxs.append([idd,bbox,int(detection.score[0]*100)])
                if draw:
                    cv2.putText(img,str(int(detection.score[0]*100))+"%",(bbox[0], bbox[1]),cv2.FONT_HERSHEY_PLAIN,2,(255,255,0),2)
                    self.fancyDraw(self.img,bbox)
        return self.img,bboxs
    def fancyDraw(self,img,bbox, l = 30, t = 4, rt = 1):
         x1, y1, w, h = bbox
         x2, y2 = x1 + w, y1 + h
         
         #Draw rectangle
         cv2.rectangle(img,bbox, (255, 0, 255),rt)
         
         #draw top left corner
         cv2.line(img,(x1,y1),(x1+l, y1),(255, 0, 255),t)
         cv2.line(img,(x1,y1),(x1, y1+l),(255, 0, 255),t)
         #draw top right corner
         cv2.line(img,(x2,y1),(x2-l, y1),(255, 0, 255),t)
         cv2.line(img,(x2,y1),(x2, y1+l),(255, 0, 255),t)
         
         #draw bottom left corner
         cv2.line(img,(x1,y2),(x1+l, y2),(255, 0, 255),t)
         cv2.line(img,(x1,y2),(x1, y2-l),(255, 0, 255),t)
         #draw bottom right corner
         cv2.line(img,(x2,y2),(x2-l, y2),(255, 0, 255),t)
         cv2.line(img,(x2,y2),(x2, y2-l),(255, 0, 255),t)
         
    
if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = FaceDetector()
    while(True):
        ret, img = capture.read()
        img,bboxs = detector.findFaces(img)
        print(bboxs)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(10, 70),cv2.FONT_HERSHEY_DUPLEX,3,(0,255,255),3)
        cv2.imshow('video original', img)
          
        if cv2.waitKey(1) == 27:
            break
      
    capture.release()
    cv2.destroyAllWindows()