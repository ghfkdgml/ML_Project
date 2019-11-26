import urllib
from konlpy.tag import Twitter
import random
import json
from bs4 import BeautifulSoup

def make_dic(words):
    tmp = ['@']
    dic = {}
    for x in words:
        tmp.append(x)
        if len(tmp)<3:continue
        if len(tmp)>3:tmp = tmp[1:]
        set_words(dic,tmp)
        if x ==".":
            tmp = ['@']
            continue
    return dic

def set_words(dic,words):
    w1,w2,w3 = words
    if not w1 in dic:dic[w1] = {}
    if not w2 in dic[w1]: dic[w1][w2]={}
    if not w3 in dic[w1][w2]:dic[w1][w2][w3]=0
    dic[w1][w2][w3] += 1

def word_choice(sel):
    keys = sel.keys()
    return random.choice(list(keys))

def make_sentence(dic):
    ret = []
    if not "@" in dic:return "check dic"
    top = dic["@"]
    w1 = word_choice(top)
    w2 = word_choice(top[w1])
    ret.append(w1)
    ret.append(w2)

    while True:
        w3 = word_choice(dic[w1][w2])
        ret.append(w3)
        if w3 =='.':break
        w1,w2 = w2,w3
    ret = "".join(ret)
    #Todo: 띄어쓰기 네이버에 등록

    # params = urllib.parse.urlencode({
    #     "_callback":"",
    #     "q":ret
    # })
    # data = urllib.request.urlopen("https://m.search.naver.com/p/csearch/dcontent/spellchecker.nhn?" + params)
    # data = data.read().decode("utf-8")[1:-2]
    # data = json.loads(data)
    # data = data["message"]['result']["html"]
    # data = soup = BeautifulSoup(data,"html.parser").getText()
    # return data
    return ret

i = 0
words = []
while True:
    if i>3:
        break
    f = open('news'+str(i)+'.txt','r',encoding='utf8')
    line = f.readline()
    twitter = Twitter()
    malist = twitter.pos(line, norm=True)
    for x in malist:
        if not x[1] in ["Punctuation"]:
            words.append(x[0])
        if x[0] ==".":
            words.append(x[0])
    i += 1
dic = make_dic(words)

print(make_sentence(dic))