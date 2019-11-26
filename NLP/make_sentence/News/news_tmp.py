#-*-coding:utf-8-*-
from konlpy.tag import Twitter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Embedding
import numpy as np
import os

Model_File = './model/model_news_v2.hdf5'
FILE_LIST = ['news0.txt', 'news1.txt', 'news2.txt', 'news3.txt']
text = []
result = []
for x in FILE_LIST:
    fp = open(x, 'r', encoding='utf-8')
    sentences = fp.readlines()
    for x in sentences:
        for y in x.split('. '):
            text.append(y.strip()+'.')
    fp.close()

twitter = Twitter()
for x in text:
    ret = twitter.pos(x, stem=True, norm=True)
    tmp = []
    for y, _ in ret:
        tmp.append(y)
    result.append(tmp)
result = np.array(result)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(result)
x_tmp = tokenizer.texts_to_sequences(result)

sen_len = 5
x_train = []
y_train = []

for x in x_tmp:
    for i in range(0,len(x)-sen_len):
        x_train.append(x[i:i+sen_len])
        y_train.append(x[i+sen_len])
x_train = np.array(x_train)
y_train = to_categorical(y_train)

vocabulary_size = len(tokenizer.word_index) + 1
embedding_dim = 30
sequence_length = x_train.shape[1]

model = Sequential()
model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length))
model.add(LSTM(128))
model.add(Dense(vocabulary_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def sample(preds, temperature=1.0):#softmax one hot -> index
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    preds = preds / np.sum(preds)
    # print(preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

if os.path.exists(Model_File):#Model data exists
    print("existing model loads!")
    try:
        model.load_weights(Model_File)
    except:
        print("model error")
    model.fit(x_train, y_train, epochs=100, verbose=2)
    test = x_train[:1]

    result = []
    i = 0
    while True:
        char = model.predict(test)
        result.append([float(np.argmax(char))])
        test = [np.append(test[:,1:],np.argmax(char))]
        test = np.array(test)
        i += 1
        if i > 50:
            break
    ttt = tokenizer.sequences_to_texts(result)
    print(ttt)

else:
    model.fit(x_train, y_train, epochs=100, verbose=2)
    model.save_weights(Model_File)
# for diversity in [1.0]:
#     generated = ''
#     sentence = ['문재인']
#     for i in range(500):
#         preds = model.predict(x, verbose=0)[0]
#         next_index = sample(preds, diversity)
#         next_char = indices_char[next_index]
#         generated += next_char
#         sentence = sentence[1:] + next_char