import numpy as np
import cv2

video_capture = cv2.VideoCapture(0)

# Nalaganje predhodno naučenega modela za zaznavanje obrazov
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Branje sličice iz video posnetka
    ret, frame = video_capture.read()

    # Pretvorba sličice v sivinsko sliko (za zaznavanje obrazov je potrebna sivinska slika)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Zaznavanje obrazov v sličici
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Označevanje zaznanih obrazov s pravokotniki
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Prikaži video posnetek z označenimi obrazi
    cv2.imshow('Video', frame)

    # Prekini zanko ob pritisku na tipko 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sprosti vir zajema video posnetka in zapri okno
video_capture.release()
cv2.destroyAllWindows()