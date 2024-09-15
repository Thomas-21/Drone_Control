import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier('tools/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('tools/haarcascade_frontalface_default.xml')

while True:
    # img = cv.imread('tools/image.jpeg')
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.15, 8)
    print(len(faces))

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 10)
        eye_gray = gray[y:y+h, x:x+w]
        eye_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(eye_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(eye_color, (ex, ey), (ex+ew, ey+eh),(0,255,0), 10)

    cv.imshow('img',frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
    
cv.destroyAllWindows()