import cv2
from keras.models import load_model
import numpy as np
import time

import dlib

class_to_label = {0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprised'}

emotion_model = load_model('./saved_models/model.h5')

IMG_SIZE = 96

# load dlib for face detection
dlib_detector = dlib.get_frontal_face_detector()

# Open a connection to the webcam
cap = cv2.VideoCapture(0)
frame_counter = 0

while True:
    
  ret, frame = cap.read()

  # detect all faces in the picture
  faces = dlib_detector(frame)

  for face in faces:
    # Extract face coordinates
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    
    # Crop and resize the face
    face_roi = frame[y:y + h, x:x + w]
    face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_roi = np.reshape(face_roi, (IMG_SIZE, IMG_SIZE, 1))
    face_roi = np.array(face_roi) / 255.0

    prediction = emotion_model.predict(np.array([face_roi]))
    
    predicted_emotion = class_to_label[np.argmax(prediction)]

    # Draw bounding box around the face
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the predicted emotion above the box
    cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

  # Display the resulting frame
  cv2.imshow('Emotion Detection', frame)

  # Break the loop if 'q' is pressed
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
