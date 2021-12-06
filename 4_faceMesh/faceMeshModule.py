import cv2
import mediapipe as mp
import time

class FacesDetector:
    def __init__(self, max_num_faces=1):
        self.max_num_faces = max_num_faces
        self.mpFaceMesh_file = mp.solutions.face_mesh
        self.fun_FaceMesh = self.mpFaceMesh_file.FaceMesh(self.max_num_faces)

        self.mpDraw_file = mp.solutions.drawing_utils
        self.draw_spec = self.mpDraw_file.DrawingSpec(color=(255,0,0),thickness = 1)
        
    def findFacesMesh(self,img,draw=True):
        self.img = img
        self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.results = self.fun_FaceMesh.process(self.imgRGB)
        self.landmarks_all = self.results.multi_face_landmarks
        faces = []
        if self.landmarks_all:
            for faceLms in self.landmarks_all:
                face = []
                for idd,lm in enumerate(faceLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    face = [idd,cx,cy]
                    if draw:
                        self.mpDraw_file.draw_landmarks(self.img, faceLms, self.mpFaceMesh_file.FACEMESH_TESSELATION,landmark_drawing_spec=self.draw_spec)
                faces.append(face)

        return self.img,faces

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = FacesDetector()
    while(True):
        ret, img = capture.read()
        img,faces = detector.findFacesMesh(img)
        

            
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(10, 70),cv2.FONT_HERSHEY_DUPLEX,3,(0,255,255),3)
        cv2.imshow('video original', img)
          
        if cv2.waitKey(1) == 27:
            break
      
    capture.release()
    cv2.destroyAllWindows()
    
    
    
    
