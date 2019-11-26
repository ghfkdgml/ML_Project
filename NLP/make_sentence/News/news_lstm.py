from bs4 import BeautifulSoup as bf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Embedding
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import pandas as pd
import random, sys, os
import pickle

Model_File = './model/model_news.hdf5'
c2i_File = './Index/char_indices.pickle'
i2c_File = './Index/indices_char.pickle'
FILE_LIST = ['news0.txt', 'news1.txt', 'news2.txt', 'news3.txt']
text = ""
for x in FILE_LIST:
    fp = open(x, 'r', encoding='utf-8')
    text += fp.readline()
    fp.close()
# chars = sorted(list(set(text)))

with open(c2i_File, 'rb') as cf:
    char_indices = pickle.load(cf)
with open(i2c_File, 'rb') as cf:
    indices_char = pickle.load(cf)

maxlen = 10
step = 3
sentences = []
next_chars = []
for i in range(0, len(text)-maxlen, step):
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])
X = np.zeros((len(sentences), maxlen, len(char_indices)), dtype=np.float32)
Y = np.zeros((len(sentences), len(char_indices)), dtype=np.float32)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    Y[i, char_indices[next_chars[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(char_indices))))
model.add(Dense(len(char_indices), activation='softmax'))
# model.add(Activation('softmax'))
# optimizer = RMSprop(lr=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model weight loads
if os.path.exists(Model_File):
    print("already existing model loads!")
    model.load_weights(Model_File)

def sample(preds, temperature=1.0):#softmax one hot -> index
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    preds = preds / np.sum(preds)
    # print(preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

model.fit(X, Y, epochs=100, verbose=2)
# for iteration in range(1, 10):
#     print()
#     print('-'*50)
#     print('반복=', iteration)
#     model.fit(X, Y, batch_size=128, nb_epoch=1, validation_data= (X,Y))
    # start_index = random.randint(0,len(text)-maxlen - 1)
start_index = 0

    # for diversity in [0.2, 0.5, 1.0, 1.2]:
for diversity in [1.0]:
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    print(sentence)
    generated += sentence
    for i in range(500):
        x = np.zeros((1,maxlen,len(char_indices)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
    with open('./result/sentence.txt','w',encoding='utf-8') as f:
        f.write(generated+'\n')
model.save_weights(Model_File)
            # sys.stdout.write(next_char+'\n')
            # sys.stdout.flush()