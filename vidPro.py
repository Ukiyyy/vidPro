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


# Funkcija, ki shrani sliko v bazo. Parameter "team" je bool, ce je iz nase mikroskupine al ne.
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


# Funkcija, ki vrne vse slike iz podatkovne baze
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
        # cv2.imshow('Face', face_image)

    # Počakajte, da uporabnik pritisne tipko
    # cv2.waitKey(0)

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


def get_piksel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1

    except:
        pass

    return new_value


def lbp_zracunan_piksel(img, x, y):
    center = img[x][y]
    value = []

    value.append(get_piksel(img, center, x - 1, y - 1))
    value.append(get_piksel(img, center, x - 1, y))
    value.append(get_piksel(img, center, x - 1, y + 1))
    value.append(get_piksel(img, center, x, y + 1))
    value.append(get_piksel(img, center, x + 1, y + 1))
    value.append(get_piksel(img, center, x + 1, y))
    value.append(get_piksel(img, center, x + 1, y - 1))
    value.append(get_piksel(img, center, x, y - 1))

    t = [1, 2, 4, 8, 16, 32, 64, 128]
    values = 0

    for i in range(len(value)):
        values += value[i] * t[i]

    return values


# Funkcija, ki izloci iz vseh slik iz baze znacilnice(lbp)
def izlocanjeZnacilnic():
    slike = dobiSlike()
    imageslbp = []
    labels = []  # Zbirka oznak

    x=0
    desired_size = (100, 100)  # Fixed size for LBP feature images

    for slika in slike:
        height, width, _ = slika.shape
        face_image = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
        img_lbp = np.zeros((height, width), np.uint8)
        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = lbp_zracunan_piksel(face_image, i, j)

        # Resize the LBP feature image to a fixed size
        img_lbp_resized = cv2.resize(img_lbp, desired_size)

        # Flatten the LBP feature image to a 1-dimensional array
        img_lbp_flattened = img_lbp_resized.flatten()

        imageslbp.append(img_lbp_flattened)

        # Dodajanje ustrezne oznake za trenutno sliko
        # Predpostavimo, da je vsaka slika povezana z razredom osebe, ki jo predstavlja
        if x%2==0:
            labels.append(1)  # Primer: 1 pomeni osebo, 0 pomeni drugo kategorijo (npr. ni oseba)
        else:
           labels.append(0)  # Represents class 0 (e.g., non-person)

        x=x+1

    return imageslbp, labels


def primerjajObraze(slika):
    slike = dobiSlike()
    imageslbp, labels = izlocanjeZnacilnic()

    # Pretvorba slik v seznam numpy array
    X = np.array(imageslbp)
    y = np.array(labels)

    # Delitev podatkov na učno in testno množico
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ustvarjanje SVM modela
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Pretvorba poslane slike v znacilnice LBP
    height, width, _ = slika.shape
    face_image = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
    img_lbp = np.zeros((height, width), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_zracunan_piksel(face_image, i, j)
    test_image = cv2.resize(img_lbp, (100, 100)).flatten()

    # Napovedovanje prepoznavanja osebe
    prediction = model.predict([test_image])

    # Preverjanje ujemanja in izpis rezultata
    if prediction in y_train:
        print("Ujema se")
    else:
        print("Ne ujema se")





# Za shranjevanje obrazov. Opcija 3 za testiranje, ce je slika v bazi
option = int(input('Enter your choice: '))
if option == 1:
    print('Dodaj novo osebo preko kamere')
    img = zaznajObrazKamera()
    shraniSliko(img, 1)
elif option == 2:
    print('Dodaj novo osebo z sliko')
    img = zaznajObrazSlika('lenna.png')
    shraniSliko(img, 0)
elif option == 3:
    slike = dobiSlike()
    cv2.imshow("Slika", slike[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif option == 4:
    slike = zaznajObrazKamera()
    primerjajObraze(slike)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cv2.waitKey()

cv2.destroyAllWindows()
