import cv2

VIDEO_FILE = 'test.mp4'
cascade_file = "C:/Users/Suho/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml"

cap = cv2.VideoCapture(VIDEO_FILE)
cv2.namedWindow('Face')
cascade = cv2.CascadeClassifier(cascade_file)