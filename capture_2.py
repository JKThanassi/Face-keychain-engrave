import numpy as np
import cv2
import random

camImg = None
cap = cv2.VideoCapture(0)
fileUID = str(random.randint(0,100000))
if not cap.isOpened():
        cap.open()

#Will need to change this from system to system        
face_cascade = cv2.CascadeClassifier('C:\\Users\\josep\\Anaconda3\\envs\\pyDev\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')

while (True):
    #capture frame by frame
    ret, frame = cap.read()
    #do facial recognition to get ROI
    #gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)


    cv2.imshow("Hit the s key to Save Picture", frame)
    if cv2.waitKey(1) == ord('s'):
        x,y,w,h = faces[0]
        roi_face = frame[y:y+h, x:x+w]
        camImg = roi_face
        cv2.imwrite("USER_PICTURE_"+str(fileUID) +".png",roi_face)
        cv2.destroyAllWindows()
        break
    elif cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break
    
    
template = cv2.imread('keychain template.png')

resized = cv2.resize(camImg,(50,50), interpolation=cv2.INTER_AREA)


for i in range(20,70):
    for j in range(110,160):
        template[i][j] = resized[i-20][j-110]

cv2.imwrite("Keychain_Template_"+str(fileUID) +".png",template)

# while not cv2.waitKey(1) == ord('s'):
#     cv2.imshow("suh", template)



#print(template.shape())


