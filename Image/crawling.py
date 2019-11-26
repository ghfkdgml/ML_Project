from bs4 import BeautifulSoup as bf
import urllib.request as req
from datetime import datetime
import re
import pickle, os

#구별할 인물
# url = "https://www.google.ru/search?q=%EB%82%98%EC%97%B0+%EC%82%AC%EC%A7%84&newwindow=1&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiv9rj5ovrkAhUVAogKHdS-CvEQ_AUIEigB&biw=1360&bih=657"
# SAVE_DIR = './img/train/main/nayeon'
#다른 인물
# url = "https://www.google.ru/search?q=%EB%8B%A4%ED%98%84+%EC%82%AC%EC%A7%84&newwindow=1&source=lnms&tbm=isch&sa=X&ved=0ahUKEwjmx6r1of3kAhXHE4gKHREiD5oQ_AUIEigB&biw=1360&bih=657"
url = "https://www.google.ru/search?q=%EC%95%84%EC%9D%B4%EC%9C%A0+%EC%82%AC%EC%A7%84&newwindow=1&source=lnms&tbm=isch&sa=X&ved=2ahUKEwj31omp6v_lAhVX7WEKHXgfA4sQ_AUoAXoECAoQAw&biw=1360&bih=657"
# SAVE_DIR = './img/train/other/other'
SAVE_DIR = './img/train/other/iu'

tmp = req.Request(url, headers={'user-agent': ':Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'})
res = req.urlopen(tmp)
soup = bf(res, "html.parser")

now = datetime.now()
# tmp = soup.select("#main")
tmp = soup.find_all('img')
# print(len(tmp))
for idx, x in enumerate(tmp[1:]):
    # t = req.urlopen(x.attrs['src']).read()
    # file_name = 'nayeon'+'_'+'%s-%s-%s' % (now.year, now.month, now.day) + '_' +str(idx)+'.jpg'
    # with open(file_name,'w') as f:
    #     f.write(t)
    # print("img save!")
    try:
        if x['data-src'] is not None:
            req.urlretrieve(x['data-src'], SAVE_DIR +'_'+'%s-%s-%s_1' % (now.year, now.month, now.day) + '_' +str(idx)+'.jpg')
    except:
        print('src:',x['src'])
        # req.urlretrieve(x['src'],'nayeon'+'_'+'%s-%s-%s' % (now.year, now.month, now.day) + '_' +str(idx)+'.jpg')
    # if x.attrs['data-src']:
    #     print(x.attrs['data-src'])