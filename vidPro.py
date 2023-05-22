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


def shraniSliko(img, team):
    slika_bin = cv2.imencode('.png', img)[1].tobytes()

    mydict = {
        "image": Binary(slika_bin),
        "isTeam": team
    }

    x = mycol.insert_one(mydict)

    if x.inserted_id:
        print("Slika uspešno shranjena v MongoDB.")
    else:
        print("Prišlo je do napake pri shranjevanju slike v MongoDB.")


def dobiSlike():
    documents = mycol.find()

    # Seznam za shranjevanje slik
    slike = []

    # Iteracija čez dokumente
    for document in documents:
        # Pridobite binarne podatke slike iz dokumenta
        image_data = document["image"]

        # Pretvorite binarne podatke v numpy array
        nparr = np.frombuffer(image_data, np.uint8)

        # Dekodirajte numpy array v sliko z uporabo OpenCV
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Dodajte sliko v seznam
        slike.append(image)

    return slike


def zaznajObrazKamera():
    video_capture = cv2.VideoCapture(0)

    # Nalaganje predhodno naučenega modela za zaznavanje obrazov
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Branje prve sličice iz video posnetka
    ret, frame = video_capture.read()

    # Pretvorba sličice v sivinsko sliko (za zaznavanje obrazov je potrebna sivinska slika)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Zaznavanje obrazov v sličici
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Izrezovanje in prikazovanje zaznanih obrazov
    for (x, y, w, h) in faces:
        # Izrez območja z obrazom iz sličice
        face_image = frame[y:y + h, x:x + w]

        # Prikaz izrezanega obraza
        #cv2.imshow('Face', face_image)

    # Počakajte, da uporabnik pritisne tipko
    #cv2.waitKey(0)

    # Sprosti vir zajema video posnetka in zapri okno
    return face_image
    video_capture.release()
    cv2.destroyAllWindows()


def zaznajObrazSlika(path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Naloži sliko
    image = cv2.imread(path)

    # Pretvori sliko v sivinsko sliko
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Zaznaj obraz na sliki
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Označi obraz z okvirjem
    for (x, y, w, h) in faces:
        face_image = image[y:y + h, x:x + w]

    # Prikaži sliko z označenimi obrazom
    return face_image
    cv2.destroyAllWindows()


"""with open("lenna.png", "rb") as f:
    image_data = f.read()

# Ustvarite slovar z binarnimi podatki slike
mydict = {
    "image": Binary(image_data),
    "isTeam": "1"
}

x = mycol.insert_one(mydict)

if x.inserted_id:
    print("Slika uspešno shranjena v MongoDB.")
else:
    print("Prišlo je do napake pri shranjevanju slike v MongoDB.")

document = mycol.find_one({"isTeam": "1"})

# Preverite, ali je dokument najden
if document:
    # Pridobite binarne podatke slike iz dokumenta
    image_data = document["image"]

    # Pretvorite binarne podatke v numpy array
    nparr = np.frombuffer(image_data, np.uint8)

    # Dekodirajte numpy array v sliko z uporabo OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Prikažite sliko z uporabo OpenCV
    cv2.imshow("Slika", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Dokument s sliko ni bil najden v MongoDB.")"""


option = int(input('Enter your choice: '))
if option == 1:
    print('Dodaj novo osebo preko kamere')
    img = zaznajObrazKamera()
    shraniSliko(img, 1)
elif option == 2:
    print('Dodaj novo osebo z sliko')
    img = zaznajObrazSlika('lenna.png')
    shraniSliko(img, 1)
elif option == 3:
    slike = dobiSlike()
    cv2.imshow("Slika", slike[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """print('Prikazi sliko iz baze')
    document = mycol.find_one({"isTeam": "0"})

    # Preverite, ali je dokument najden
    if document:
        # Pridobite binarne podatke slike iz dokumenta
        image_data = document["image"]

        # Pretvorite binarne podatke v numpy array
        nparr = np.frombuffer(image_data, np.uint8)

        # Dekodirajte numpy array v sliko z uporabo OpenCV
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Prikažite sliko z uporabo OpenCV
        cv2.imshow("Slika", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Dokument s sliko ni bil najden v MongoDB.")"""
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
