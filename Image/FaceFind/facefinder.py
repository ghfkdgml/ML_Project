import cv2
from PIL import Image
import glob


image_file = '../img/train/main/*.jpg'
# other_file = '../img/train/other/*.jpg'
other_file = '../img/train/other/iu_2019-11-23_1_*'
result_path = './result/'
other_path = './other/'
cascade_file = "C:/Users/Suho/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml"

X = []
Y = []
# files = glob.glob(image_file)
files = glob.glob(other_file)
i = 400
print(len(files))
for f in files:
    image = cv2.imread(f)
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_file)
    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1,
                                         minNeighbors=3,
                                         minSize=(10, 10),
                                         maxSize=(150, 150))
    if len(face_list) > 0:
        for face in face_list:
            x, y, w, h = face
            # face_img = image[y:y+h, x:x+w]
            face_img = image_gs[y:y + h, x:x + w]
            # face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2)
            face_img = cv2.resize(face_img, (150, 150))
            # face_img = face_img.astype('float32') / 255
            # result_img = Image.fromarray(face_img)#cv2 img -> pil img
            # cv2.imwrite(result_path + str(i) + ".PNG", face_img)
            cv2.imwrite(other_path + str(i) + ".PNG", face_img)
            i += 1

    else:
        print("no face")
        continue