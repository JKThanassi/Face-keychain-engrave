import numpy as np
import cv2
import random

camImg = None
cap = cv2.VideoCapture(0)
fileUID = str(random.randint(0,100000))
if not cap.isOpened():
        cap.open()

# Will need to change this from system to system
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
    
    
template = cv2.imread('new_template.png')

resized = cv2.resize(camImg,(200,200), interpolation=cv2.INTER_AREA)
# resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


for i in range(200,400):
    for j in range(525,725):
        template[i][j] = resized[i-200][j-525]

cv2.imwrite("Keychain_Template_"+str(fileUID) +".png",template)

# while not cv2.waitKey(1) == ord('s'):
#     cv2.imshow("suh", template)






