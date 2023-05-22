import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import os
from os import listdir
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
skupina = 1
drugo = 0


def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1

    except:
        pass

    return new_value


def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []

    val_ar.append(get_pixel(img, center, x - 1, y - 1))
    val_ar.append(get_pixel(img, center, x - 1, y))
    val_ar.append(get_pixel(img, center, x - 1, y + 1))
    val_ar.append(get_pixel(img, center, x, y + 1))
    val_ar.append(get_pixel(img, center, x + 1, y + 1))
    val_ar.append(get_pixel(img, center, x + 1, y))
    val_ar.append(get_pixel(img, center, x + 1, y - 1))
    val_ar.append(get_pixel(img, center, x, y - 1))

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val


# Funkcija za izračun LBP značilnic
def calculate_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


# Branje in predobdelava podatkov
def preprocess_data():
    imageslbp = []
    labels = []
    for images in os.listdir("C:/Users/urosm/PycharmProjects/vidPro"):
        if (images.endswith(".png")):
            img = cv2.imread(images)
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                face_image = img[y:y + h, x:x + w]
                print(face_image.shape)
                height, width, _ = face_image.shape
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                img_lbp = np.zeros((height,width),np.uint8)
                for i in range(0, height):
                    for j in range(0, width):
                        img_lbp[i, j] = lbp_calculated_pixel(face_image, i, j)

    cv2.waitKey()
    return images, labels


preprocess_data()
cv2.waitKey()

cv2.destroyAllWindows()
