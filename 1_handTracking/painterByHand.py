import cv2
import time
import handTrackingModule as htm
import os
import numpy as np
capture = cv2.VideoCapture(0)
capture.set(3,1280)
capture.set(4,720)



ImagesPath = 'virual_painter_images'

imgList = os.listdir(ImagesPath)

ImagesList = []
for im in imgList:
    image = cv2.imread(f'{ImagesPath}/{im}')
    image = image[0:150,0:]
    ImagesList.append(image)
print((len(ImagesList)))
pTime = 0
cTime = 0
detector = htm.HandsDetector(num_hands=4,detection_con=0.85)
colors = {'red':(0,0,255),'green':(0,255,0),'blue':(255,0,0),'orange':(0,165,255),'light_yellow':(102,255,255),
          'yellow':(0,255,255),'pink':(255,0,250),'light_green':(144,238,144),'eraser':(0,0,0)}

imgCanvas = np.zeros((720,1280,3),np.uint8)
selectedColor = 'red'
imageIndex = 0
brushThickness = 15
eraserThickness = 50
xp,yp = 0, 0
while(True):
    ret, img = capture.read()
    img = cv2.flip(img, 1)
    
    img = detector.findHands(img)
    lms_list = detector.findPositions(img, draw = False)
    if len(lms_list) !=0:
        x1, y1 = lms_list[8][1:]
        x2, y2 = lms_list[12][1:]
        fingers = detector.fingersUp(ltr = False)
        #fingers selected
        if fingers[0]==0 and fingers[1]==1 and fingers[2] == 1 and fingers[3]==0 and fingers[4]==0: 
            cv2.rectangle(img, (x1,y1-25), (x2,y2+25), colors[selectedColor], cv2.FILLED)
            xp,yp = 0, 0
            if y1 < 150:
                #Red
                if 0<x1<100:
                    selectedColor = 'red'
                    imageIndex = 0
                elif 135<x1<270:
                    selectedColor = 'green'
                    imageIndex = 1
                elif 270<x1<405:
                    selectedColor = 'blue'
                    imageIndex = 2
                
                elif 405<x1<540:
                    selectedColor = 'orange'
                    imageIndex = 3
                elif 540<x1<675:
                    selectedColor = 'light_yellow'
                    imageIndex = 4
                elif 675<x1<810:
                    selectedColor = 'yellow'
                    imageIndex = 5
                    
                elif 810<x1<945:
                    selectedColor = 'pink'
                    imageIndex = 6
                elif 945<x1<1080:
                    selectedColor = 'light_green'
                    imageIndex = 7
                else:
                    selectedColor = 'eraser'
                    imageIndex = 8
                    
                    
            print("Selction")
             
        elif fingers[0]==0 and fingers[1]==1 and fingers[2] == 0 and fingers[3]==0 and fingers[4]==0:
            cv2.circle(img,(x1,y1),10,colors[selectedColor],cv2.FILLED)
            
            print("painting")
            if xp ==0 and yp == 0:
                xp, yp = x1, y1
                
            if selectedColor == 'eraser':
                cv2.line(img,(xp,yp),(x1, y1),colors[selectedColor],eraserThickness)
                cv2.line(imgCanvas,(xp,yp),(x1, y1),colors[selectedColor],eraserThickness)
            else:
                cv2.line(img,(xp,yp),(x1, y1),colors[selectedColor],brushThickness)
                cv2.line(imgCanvas,(xp,yp),(x1, y1),colors[selectedColor],brushThickness)
            xp, yp = x1, y1
        elif fingers[0]==1 and fingers[1]==1 and fingers[2] == 1 and fingers[3]==1 and fingers[4]==1: 
            imgCanvas [:,:] = (0,0,0)

    grayImg = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)    
    _ , imgInv = cv2.threshold(grayImg, 50,255, cv2.THRESH_BINARY_INV) 
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    print(imgInv.shape)
    print(img.shape)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10, 70),cv2.FONT_HERSHEY_DUPLEX,3,(0,255,255),3)
    img[0:150,0:] = ImagesList[imageIndex]
    cv2.imshow('video original', img)
    
    if cv2.waitKey(1) == 27:
        break
  
capture.release()
cv2.destroyAllWindows()