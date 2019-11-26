from keras.models import Model, Sequential
from keras.layers import Dense, Concatenate, Dropout, LSTM, Embedding,  Input, Reshape, Conv2D, MaxPool2D, Flatten
from keras.optimizers import Adam
from keras import regularizers
from keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Twitter
from sklearn.model_selection import train_test_split
import numpy as np
import pickle, os


c2i_File = './Index/char_indices.pickle'
i2c_File = './Index/indices_char.pickle'
Model_File = './model/model_news.hdf5'

with open(c2i_File, 'rb') as cf:
    char_indices = pickle.load(cf)
with open(i2c_File, 'rb') as cf:
    indices_char = pickle.load(cf)
print(char_indices)
FILE_LIST = ['novel.txt']
c2i_File = './Index/char_indices.pickle'
i2c_File = './Index/indices_char.pickle'
max_len = 10
text = []
X = []#train data
Y = []#label data
twitter = Twitter()
for x in FILE_LIST:
    fp = open(x, 'r')
    sentences = fp.readlines()
    for sentence in sentences:
        ret = twitter.pos(sentence)
        for a in ret:
            try:
                text.append(char_indices[a[0]])
            except:
                print(a[0])
    fp.close()

for x in range(len(text) - max_len):
    X.append(text[x:x+max_len])
    Y.append(text[x+max_len])
# X_tmp = np.zeros((len(X), max_len, len(char_indices)))
X_tmp = np.array(X)
Y_tmp = np.zeros((len(Y), len(char_indices)))
for i, sentence in enumerate(X):
    # for t in range(max_len):
    #     X_tmp[i, t, sentence[t]] = 1
    Y_tmp[i, Y[i]] = 1

x_train, x_test, y_train, y_test = train_test_split(X_tmp,Y_tmp, test_size=.2, random_state=777)

model = Sequential()
model.add(Embedding(len(char_indices), 256))
model.add(LSTM(128, input_shape=(max_len, len(char_indices)), kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(len(char_indices), activation='softmax', kernel_regularizer=regularizers.l2(0.01)))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# sequence_length = x_train.shape[1]
# vocabulary_size = len(char_indices)
# embedding_dim = 256
# filter_sizes = [3,4,5]
# num_filters = 512
# drop = 0.5
#
# epochs = 10
# batch_size = 30

# inputs = Input(shape=(sequence_length,), dtype='int32')
# embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
# reshape = Reshape((sequence_length,embedding_dim,1))(embedding)
#
# conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
# conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
# conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
#
# maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
# maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
# maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
#
# concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
# flatten = Flatten()(concatenated_tensor)
# dropout = Dropout(drop)(flatten)
# output = Dense(units=len(char_indices), activation='softmax')(dropout)
# model = Model(inputs=inputs, outputs=output)
# adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#
# model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

if os.path.exists(Model_File):
    # print("already existing model loads!")
    # model.load_weights(Model_File)
    model.fit(x_train, y_train, epochs=10, verbose=2, validation_data=(x_test, y_test))
    # model.fit(X, Y, epochs=10, verbose=2)
else:
    model.fit(x_train, y_train, epochs=10, verbose=2, validation_data=(x_test, y_test))

def sample(preds, temperature=1.0):#softmax one hot -> index
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    preds = preds / np.sum(preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# print(model.evaluate(x_test,y_test))
# gen = ""
# ttt = x_test[1:2]
# for x in ttt[0]:
#     idx = np.argmax(x)
#     gen += indices_char[idx]
#
# i = 0
# while True:
#     if i > 20:
#         break
#     prediction = model.predict(ttt, verbose=0)
#     next_index = sample(prediction)
#     # if next_index == 0:
#     #     break
#     next_char = indices_char[next_index]
#     gen += next_char
#     ttt[0] = np.concatenate((ttt[0][:-1], [np.eye(3450)[1]]))
#     i += 1
#
# print(gen)
model.save_weights(Model_File)