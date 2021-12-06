import cv2
import mediapipe as mp
import time
capture = cv2.VideoCapture(0)

mpFaceMesh_file = mp.solutions.face_mesh
fun_FaceMesh = mpFaceMesh_file.FaceMesh(max_num_faces=1)

mpDraw_file = mp.solutions.drawing_utils
draw_spec = mpDraw_file.DrawingSpec(color=(255,0,0),thickness = 1)

pTime = 0
cTime = 0
while(True):
    ret, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = fun_FaceMesh.process(imgRGB)
    
    landmarks_all = results.multi_face_landmarks
    
    for faceLms in landmarks_all:
        for idd,lm in enumerate(faceLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(idd,cx,cy)
        mpDraw_file.draw_landmarks(img, faceLms, mpFaceMesh_file.FACEMESH_TESSELATION,landmark_drawing_spec=draw_spec)     
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10, 70),cv2.FONT_HERSHEY_DUPLEX,3,(0,255,255),3)
    cv2.imshow('video original', img)
      
    if cv2.waitKey(1) == 27:
        break
  
capture.release()
cv2.destroyAllWindows()