import numpy as np
from konlpy.tag import Twitter
import re, os

twit = Twitter()
GET_TYPE = ['Noun', 'Verb', 'Adjective']
c2i_File = './Index/word.npy'
train = []

with open('train_data', encoding='utf-8') as f:
    sentences = f.readlines()
    i = 0
    # for x in sentences[:1000000]:
    for x in sentences:
        if i % 100000 == 0:
            print(i)
        sentence = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-z ,]", "", x)#only Korean Lang, English
        words = twit.pos(x, stem=True, norm=True)
        tmp = []

        for word,t in words:
            if t in GET_TYPE:
                tmp.append(word)
            if t == 'Alpha':
                tmp.append(word.lower())
        if len(tmp) == 0:#ex) ㅠㅠ, ㅋㅋ
            for word, _ in words:
                tmp.append(word)

        train.append(tmp)
        i += 1

train = np.array(train)

if os.path.exists(c2i_File):
    before_train = np.load(c2i_File, allow_pickle=True)
    train = np.concatenate((before_train, train))
    np.save(c2i_File, train)
else:
    np.save(c2i_File, train)
