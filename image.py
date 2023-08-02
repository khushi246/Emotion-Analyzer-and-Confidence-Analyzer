import cv2
from deepface import DeepFace
import numpy as np

face_cascade= cv2.CascadeClassifier("C:\\Users\\Khushi Vishwakarma\\AppData\\Roaming\\Python\\Python311\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
imgpath ="D:\emotion_analysis\happy_emotion.jpeg"
image = cv2.imread(imgpath)
analyze = DeepFace.analyze(image,actions=['emotion'])
print(analyze)


