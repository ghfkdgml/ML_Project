import cv2

# image_file = './img/me/2019-02-16-14-50-30.jpg'
image_file = './img/train/main/nayeon_2019-10-2_22.jpg'
cascade_file = "C:/Users/Suho/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml"

image = cv2.imread(image_file)
image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cascade = cv2.CascadeClassifier(cascade_file)
face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1,
                                     minNeighbors=3,
                                     minSize=(70,70))
print(face_list)

if len(face_list)>0:
    color = (0,0,255)
    for face in face_list:
        x,y,w,h = face
        cv2.rectangle(image, (x,y), (x+w,y+h),color,thickness=3)
    cv2.imwrite("facedetect-ouput.PNG", image)
else:
    print("No face")