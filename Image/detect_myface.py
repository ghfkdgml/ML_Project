import cv2
import os
from PIL import Image
import glob
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np

Model_File = 'model.hdf5'
image_file = './img/train/main/*.jpg'
other_file = './img/train/other/*.jpg'
test_file = './img/test/*.jpg'
tmp_file = './img/me/2018-01-06-15-28-38.jpg'
cascade_file = "C:/Users/Suho/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml"

X = [] #image Data
Y = [] #Label Data

#구별하고자 하는 사람 데이터
files = glob.glob(image_file)
i = 0
for f in files:
    LABEL = [1, 0, 0]
    image = cv2.imread(f)
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_file)
    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1,
                                         minNeighbors=3,
                                         minSize=(30, 30),
                                         maxSize=(100, 100))
    if len(face_list) > 0:
        for face in face_list:
            x, y, w, h = face
            color = (0, 0, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=3)
            cv2.imwrite(str(i)+".PNG", image)
            i += 1
            face_img = image[y:y+h, x:x+w]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (150, 150))
            face_img = face_img.astype('float32') / 255
            height, width, channel = face_img.shape
            X.append(face_img)
            Y.append(LABEL)
            # result_img = Image.fromarray(face_img)#cv2 img -> pil img
            for ang in range(-20, 20, 5):
                matrix = cv2.getRotationMatrix2D((width/2, height/2), ang, 1)
                dst = cv2.warpAffine(face_img, matrix, (width, height))
                X.append(dst/256)
                Y.append(LABEL)
                trans_pic1 = cv2.flip(face_img,1)
                X.append(trans_pic1/256)
                Y.append(LABEL)
                trans_pic2 = cv2.flip(dst, 1)
                X.append(trans_pic2/256)
                Y.append(LABEL)
    else:
        print("no face")
        continue

others = glob.glob(other_file)
for f in others:
    LABEL = [0, 1, 0]
    image = cv2.imread(f)
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_file)
    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1,
                                         minNeighbors=3,
                                         minSize=(30, 30),
                                         maxSize=(100, 100))
    if len(face_list) > 0:
        for face in face_list:
            x, y, w, h = face
            face_img = image[y:y+h, x:x+w]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (150, 150))
            face_img = face_img.astype('float32') / 255
            height, width, channel = face_img.shape
            X.append(face_img)
            Y.append(LABEL)
            # result_img = Image.fromarray(face_img)#cv2 img -> pil img
            for ang in range(-20, 20, 5):
                matrix = cv2.getRotationMatrix2D((width/2, height/2), ang, 1)
                dst = cv2.warpAffine(face_img, matrix, (width, height))
                X.append(dst/256)
                Y.append(LABEL)
                trans_pic1 = cv2.flip(face_img,1)
                X.append(trans_pic1/256)
                Y.append(LABEL)
                trans_pic2 = cv2.flip(dst, 1)
                X.append(trans_pic2/256)
                Y.append(LABEL)
    else:
        print("no other face")
        continue

def build_model(input_shape):
    model = Sequential()
    model.add(Convolution2D(32,3,3,
                            border_mode='same',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3,
                            border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop')
    return model

X = np.array(X)
Y = np.array(Y)
model = build_model(X[0].shape)
# model.fit(X, Y, batch_size=128, nb_epoch=30)
if os.path.exists(Model_File):
    print("model load")
    model.load_weights(Model_File)

    TestList = [tmp_file]

    for f in TestList:
        image = cv2.imread(f)
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cascade_file)
        face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1,
                                             minNeighbors=3,
                                             minSize=(30, 30),
                                             maxSize=(70, 70))
        Test = []
        if len(face_list) > 0:
            for face in face_list:
                x, y, w, h = face
                face_img = image[y:y+h, x:x+w]
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img = cv2.resize(face_img, (150, 150))
                face_img = face_img.astype('float32') / 255
                Test.append(face_img)
                Test = np.array(Test)
                print(model.predict(Test))
                print(np.argmax(model.predict(Test)))

else:
    model.fit(X, Y, nb_epoch=2)
    X_test = X[0:3]
    for x in model.predict(X_test):
        print(np.argmax(x))
    model.save_weights(Model_File)