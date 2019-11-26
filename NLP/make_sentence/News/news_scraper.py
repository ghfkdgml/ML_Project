from bs4 import BeautifulSoup as bf
import urllib.request as req
import re
import pickle, os

url = "https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=100"

res = req.urlopen(url)
soup = bf(res, "html.parser")

def clean_text(text):
    cleaned_text = re.sub('[a-zA-Z]', '', text)
    cleaned_text = re.sub('[\{\}\[\]\/?,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]',
                          '', cleaned_text)
    return cleaned_text



tmp = soup.select("#main_content .list_body .cluster a")
i = 0
for x in tmp[3:]:
    if i>3:
        break
    url = x.attrs['href']
    if not url.startswith('http'):
        continue
    response = req.urlopen(url)
    soup_article = bf(response,"html.parser")
    content = soup_article.select("#articleBodyContents")
    text = content[0].getText().split('\n')
    for x in range(len(text)):
        if len(text[x])>50:
            with open('news'+str(i)+'.txt','a',encoding='utf-8') as f:
                f.write(clean_text(text[x]))
                f.write('\n')
    i+=1


# sentence -> char vector

c2i_File = './Index/char_indices.pickle'
i2c_File = './Index/indices_char.pickle'
FILE_LIST = ['news0.txt','news1.txt','news2.txt','news3.txt']
text = ""
for x in FILE_LIST:
    fp = open(x, 'r', encoding='utf-8')
    text += fp.readline()
    fp.close()
chars = sorted(list(set(text)))
if os.path.exists(c2i_File):
    cf = open(c2i_File, 'rb')
    char_indices = pickle.load(cf)
    for x in chars:
        if x not in char_indices.keys():
            char_indices[x] = len(char_indices)
    cf = open(c2i_File, 'wb')
    pickle.dump(char_indices, cf, pickle.HIGHEST_PROTOCOL)
else:
    char_indices = dict((c, i) for i, c in enumerate(chars))
    cf = open(c2i_File, 'wb')
    pickle.dump(char_indices, cf, pickle.HIGHEST_PROTOCOL)

if os.path.exists(i2c_File):
    indf = open(i2c_File, 'rb')
    indices_char = pickle.load(indf)
    for x in chars:
        if x not in indices_char.values():
            indices_char[len(indices_char)] = x
    indf = open(i2c_File, 'wb')
    pickle.dump(indices_char, indf, pickle.HIGHEST_PROTOCOL)
else:
    indices_char = dict((i, c) for i, c in enumerate(chars))
    indf = open(i2c_File, 'wb')
    pickle.dump(indices_char, indf, pickle.HIGHEST_PROTOCOL)

print('char_indices: ',len(char_indices))
print('indices_char: ',len(indices_char))