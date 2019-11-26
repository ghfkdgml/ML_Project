import cv2
import os
from PIL import Image
import glob
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
import numpy as np

Model_File = 'model_2.hdf5'
image_file = './result/*.PNG'
other_file = './other/*.PNG'
# tmp_file = '../img/me/2018-01-06-15-28-38.jpg'
tmp_file = './test/test.jpg'
cascade_file = "C:/Users/Suho/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml"

files = glob.glob(image_file)
X = []
Y = []

#preprocessing
for f in files:
    LABEL = [0, 1]
    image = cv2.imread(f)
    image = image.astype('float32') / 255 #normalization
    X.append(image)
    Y.append(LABEL)
    height, width, channel = image.shape
    for ang in range(-20, 20, 5):
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), ang, 1)
        dst = cv2.warpAffine(image, matrix, (width, height))
        X.append(dst / 256)
        Y.append(LABEL)
        trans_pic1 = cv2.flip(image, 1)
        X.append(trans_pic1 / 256)
        Y.append(LABEL)
        trans_pic2 = cv2.flip(dst, 1)
        X.append(trans_pic2 / 256)
        Y.append(LABEL)

files = glob.glob(other_file)
for f in files:
    LABEL = [1, 0]
    image = cv2.imread(f)
    image = image.astype('float32') / 255 #normalization
    X.append(image)
    Y.append(LABEL)
    height, width, channel = image.shape
    for ang in range(-20, 20, 5):
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), ang, 1)
        dst = cv2.warpAffine(image, matrix, (width, height))
        X.append(dst / 256)
        Y.append(LABEL)
        trans_pic1 = cv2.flip(image, 1)
        X.append(trans_pic1 / 256)
        Y.append(LABEL)
        trans_pic2 = cv2.flip(dst, 1)
        X.append(trans_pic2 / 256)
        Y.append(LABEL)

def build_model(input_shape):
    # model = Sequential()
    # model.add(Convolution2D(32,3,3,
    #                         border_mode='same',
    #                         input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Convolution2D(64, 3, 3,
    #                         border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(64, 3, 3))
    # model.add(MaxPool2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Flatten())
    # model.add(Dense(512))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    #
    # model.add(Dense(2))
    # model.add(Activation('softmax'))
    # model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'])

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

X = np.array(X)
Y = np.array(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = build_model(X[0].shape)

# if os.path.exists(Model_File):
#     print("model load")
#     model.load_weights(Model_File)
#
#     TestList = [tmp_file]
#
#     for f in TestList:
#         image = cv2.imread(f)
#         image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         cascade = cv2.CascadeClassifier(cascade_file)
#         face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1,
#                                              minNeighbors=3,
#                                              minSize=(30, 30),
#                                              maxSize=(200, 200))
#         Test = []
#         test_face = []
#         j = 0
#         if len(face_list) > 0:
#             for face in face_list:
#                 x, y, w, h = face
#                 test_face.append([x,y,w,h])
#                 face_img = image[y:y+h, x:x+w]
#                 face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
#                 face_img = cv2.resize(face_img, (150, 150))
#                 face_img = face_img.astype('float32') / 255
#                 Test.append([face_img])
#             Test = np.array(Test)
#             for tmp in Test:
#                 x, y, w, h = test_face[j]
#                 if np.argmax(model.predict(tmp)) == 1:
#                     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
#                 else:
#                     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness=3)
#                 j += 1
#             cv2.imwrite("facedetect-output.PNG", image)


if os.path.exists(Model_File):
    print("model load")
    model.load_weights(Model_File)
    model.fit(X_train, y_train, nb_epoch=2, validation_data=(X_test, y_test))
    model.save_weights(Model_File)
