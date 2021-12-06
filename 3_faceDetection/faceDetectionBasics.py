import cv2
import mediapipe as mp
import time
capture = cv2.VideoCapture(0)

mpFaceDetection_file = mp.solutions.face_detection
fun_face_detection = mpFaceDetection_file.FaceDetection()

mpDraw_file = mp.solutions.drawing_utils

pTime = 0
cTime = 0
while(True):
    ret, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = fun_face_detection.process(imgRGB)
    
    detections_all = results.detections
    
    if detections_all:
        for idd,detection in enumerate(detections_all):
            #mpDraw_file.draw_detection(img,detection)
            bboxc = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih),\
                int(bboxc.width * iw), int(bboxc.height * ih)
            
            cv2.putText(img,str(int(detection.score[0]*100))+"%",(bbox[0], bbox[1]),cv2.FONT_HERSHEY_PLAIN,2,(255,255,0),2)
            cv2.rectangle(img, bbox, (255,0,255),2)
            #print(idd,detection.location_data.relative_bounding_box)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10, 70),cv2.FONT_HERSHEY_DUPLEX,3,(0,255,255),3)
    cv2.imshow('video original', img)
      
    if cv2.waitKey(1) == 27:
        break
  
capture.release()
cv2.destroyAllWindows()