import cv2
import numpy as np
from keras.models import model_from_json


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# start the webcam feed
#cap = cv2.VideoCapture(0)

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
#cap = cv2.VideoCapture("C:\\Users\\Khushi Vishwakarma\\Downloads\\pexels-artem-podrez-6952269-3840x2160-30fps.qmp4")
cap = cv2.VideoCapture("C:\\Users\\Khushi Vishwakarma\\Downloads\\pexels-artem-podrez-6952257-3840x2160-30fps.mp4")


#new code for define new function of calculating most
def most_frequent(emotion_dict_store):
    counter = 0
    num = emotion_dict_store[0]
    for i in emotion_dict_store:
        curr_frequency = emotion_dict_store.count(i)
        if(curr_frequency > counter):
            counter = curr_frequency
            num = i
    return num

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)



        emotion_dict_store = []
        emotion_dict_store.append(emotion_dict[maxindex]) 
    
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print('dominant emotion detected:-')
print(most_frequent(emotion_dict_store))
confidence=most_frequent(emotion_dict_store)
if confidence == "Happy":
    print('You are confident enough')
elif confidence == "Neutral":
    print('you need to work on your confidence')
else:
    print('you are not confident enough')
        
cap.release()
cv2.destroyAllWindows()
