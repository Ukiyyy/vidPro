import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Funkcija za izračun LBP značilnic
def calculate_lbp(image):
    # Pretvorba slike v sivinsko območje
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Izračun LBP značilnic
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# Branje in predobdelava podatkov
def preprocess_data():
    # Branje slik in pridobivanje LBP značilnic ter oznak
    images = []
    labels = []
    for i in range(num_samples):
        image = cv2.imread(f"path/to/images/{i}.jpg")
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_image = image[y:y+h, x:x+w]
            lbp_features = calculate_lbp(face_image)
            images.append(lbp_features)
            labels.append(class_label)  # Nastavite ustrezno oznako glede na razpoznavni razred
    return images, labels

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