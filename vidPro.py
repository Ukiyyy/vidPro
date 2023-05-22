import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import os
import pymongo
from bson.binary import Binary
from os import listdir
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


myclient = pymongo.MongoClient("mongodb+srv://vidPro:vidPro@vidpro.wsmmizs.mongodb.net/")
mydb = myclient["vidPro"]
mycol = mydb["images"]

with open("lenna.png", "rb") as f:
    image_data = f.read()

# Ustvarite slovar z binarnimi podatki slike
mydict = {
    "image": Binary(image_data),
    "isTeam": "1"
}
#mydict = { "image": "John", "isTeam": "1" }

x = mycol.insert_one(mydict)

if x.inserted_id:
    print("Slika uspešno shranjena v MongoDB.")
else:
    print("Prišlo je do napake pri shranjevanju slike v MongoDB.")

"""face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
                img_lbp = np.zeros((height, width), np.uint8)
                for i in range(0, height):
                    for j in range(0, width):
                        img_lbp[i, j] = lbp_calculated_pixel(face_image, i, j)
                imageslbp.append(img_lbp)

    print(imageslbp)
    cv2.imshow("gds", imageslbp[0])
    cv2.waitKey()
    return images, labels


preprocess_data()
cv2.waitKey()

cv2.destroyAllWindows()"""
