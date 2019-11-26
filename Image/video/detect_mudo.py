import cv2

VIDEO_FILE = 'test.mp4'
Result = 'result.mp4'
cascade_file = "C:/Users/Suho/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml"

cap = cv2.VideoCapture(VIDEO_FILE)
# cv2.namedWindow('Face')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(Result, fourcc, fps, (int(width), int(height)))

cascade = cv2.CascadeClassifier(cascade_file)
while(True):
    ret, frame = cap.read()
    if frame is None:
        print("frame is null")
        break

    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_list = cascade.detectMultiScale(grayframe, scaleFactor=1.1,
                                         minNeighbors=3,
                                         minSize=(30, 30))
    for (x, y, w, h) in face_list:
        color = (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=3)
        print(x, y, w, h, "line drawing")

    # cv2.imshow('Face', frame)
    out.write(frame)
cap.release()
# cv2.destroyAllWindows('Face')
