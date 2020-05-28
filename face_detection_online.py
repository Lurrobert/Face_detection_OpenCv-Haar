import cv2
import sys
from mtcnn.mtcnn import MTCNN

video_capture = cv2.VideoCapture(0)
# Haar
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    detector = MTCNN()
    #boxes = detector.detect_faces(frame)
    boxes = classifier.detectMultiScale(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for box in boxes:
        x, y, w, h = box   # ['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
