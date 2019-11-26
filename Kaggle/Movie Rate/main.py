from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, LSTM
from keras.layers import Reshape, Flatten, Dropout, Concatenate, Bidirectional
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import re
import os

data_all = './Index/word_100.npy'
Model_File = 'model_test.hdf5'
max_len = 40

tokenizer = Tokenizer()
train = np.load(data_all, allow_pickle=True)

label = []
with open('train_label', encoding='utf-8') as f:
    sentences = f.readlines()
    for x in sentences[:6000000]:
        x = re.sub('\n', '', x)
        label.append([x])

tokenizer.fit_on_texts(train)
x_train = tokenizer.texts_to_sequences(train)
X = pad_sequences(x_train, maxlen=max_len)

X = np.array(X)
Y = to_categorical(label, 11)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

sequence_length = X_train.shape[1]
vocabulary_size = len(tokenizer.word_index) + 1
embedding_dim = 256
filter_sizes = [3, 4, 5]
num_filters = 256
drop = 0.5
epochs = 1
batch_size = 1024

#Model
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
reshape = Bidirectional(LSTM(128, dropout=0.5, return_sequences=True))(embedding)
reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
dense = Dense(256, activation='relu')(dropout)
dropout = Dropout(drop)(dense)
output = Dense(units=11, activation='softmax')(dropout)

model = Model(inputs=inputs, outputs=output)
adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])

#Test data load
final_test = np.load('./Index/test_data.npy', allow_pickle=True)
final_test = tokenizer.texts_to_sequences(final_test)
Final = pad_sequences(final_test, maxlen=max_len)
test_id = [x for x in range(400000)]
result = []

if os.path.exists(Model_File):#Model data exists
    print("existing model loads!")
    model.load_weights(Model_File)
    prediction = model.predict(Final)
    print('save start!')
    for x in range(400000):
        result.append(np.argmax(prediction[x]))
    submission = pd.DataFrame({
        "ID": test_id,
        "Prediction": result
    })
    submission.to_csv('submission6.csv', index=False)
else:
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))



model.save_weights(Model_File)