from konlpy.tag import Twitter
from collections import Counter
import pickle, os
import matplotlib.pyplot as plt

cnt_word = Counter()
FILE_LIST = ['novel.txt']
c2i_File = './Index/char_indices.pickle'
i2c_File = './Index/indices_char.pickle'
max_len = 10
len2sentence = []
text = ""
X = []
word_set = set()
twitter = Twitter()
for x in FILE_LIST:
    fp = open(x, 'r')
    sentences = fp.readlines()
    for sentence in sentences:
        text += sentence
        ret = twitter.pos(sentence)
        len2sentence.append(len(ret))
        for a in ret:
            word_set.add(a[0])
            cnt_word[a[0].lower()] = cnt_word[a[0].lower()] + 1
    # for x in range()
    fp.close()

word_sorted = sorted(cnt_word.items(), key=lambda x:x[1], reverse=True)
word2idx = {}
idx2word = {}
i = 0
for word, _ in word_sorted:
    word2idx[word] = i
    idx2word[i] = word
    i += 1
print(word2idx)
print(idx2word)

if os.path.exists(c2i_File):
    cf = open(c2i_File, 'rb')
    char_indices = pickle.load(cf)
    if len(char_indices) != len(word2idx):
        cf = open(c2i_File, 'wb')
        pickle.dump(word2idx, cf, pickle.HIGHEST_PROTOCOL)
else:
    cf = open(c2i_File, 'wb')
    pickle.dump(word2idx, cf, pickle.HIGHEST_PROTOCOL)

if os.path.exists(i2c_File):
    indf = open(i2c_File, 'rb')
    indices_char = pickle.load(indf)
    if len(indices_char) != len(idx2word):
        indf = open(i2c_File, 'wb')
        pickle.dump(idx2word, indf, pickle.HIGHEST_PROTOCOL)
else:
    indf = open(i2c_File, 'wb')
    pickle.dump(idx2word, indf, pickle.HIGHEST_PROTOCOL)

# max len:200, average len: 20
# plt.hist(len2sentence, bins=50)
# plt.show()
