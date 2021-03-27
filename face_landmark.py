import numpy as np
import cv2 as cv
import dlib

cap = cv.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor  = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    _ , frame = cap.read()
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        #cv.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),3)

        landmarks = predictor(gray, face)

        for n in range (0,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv.circle(frame,(x,y),2,(0,255,0),-1)
        
        
    # Display the resulting frame
    cv.imshow('frame', frame)





    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

print(".........Ended.........")