import cv2
from keras.models import load_model
import numpy as np
import time

import dlib


def prediction_percent_str(prediction):
  return str(prediction[0]) + ": " + "{:.2f}".format(prediction[1] * 100) + "%"

def emotion_to_color(emotion):
    color_mapping = {
        'angry': (0, 0, 255),    # Red
        'happy': (0, 255, 0),    # Green
        'sad': (255, 0, 0),      # Blue
        'surprised': (255, 255, 0),  # Yellow
        'neutral': (128, 128, 128),  # Gray
        'fearful': (255, 255, 255), # White
    }
    return color_mapping.get(emotion, (255, 255, 255))



class_to_label = {0: 'angry', 1: 'fearful', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprised'}

emotion_model = load_model('../saved_models/basicCNN/basicCNN_50_32-1.h5')

IMG_SIZE = 96

# load dlib for face detection
dlib_detector = dlib.get_frontal_face_detector()

# Open a connection to the webcam
cap = cv2.VideoCapture(0)
frame_counter = 0

def predict_emotions():
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
    
    top_n_indices = np.argsort(prediction[0])[::-1][:3]
    # Retrieve the top N predictions and their corresponding probabilities
    top_n_predictions = [(class_to_label[i], prediction[0][i]) for i in top_n_indices]
    predicted_emotion = class_to_label[np.argmax(prediction)]

    # Draw bounding box around the face
    cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_to_color(top_n_predictions[0][0]), 2)

    # Display the predicted emotion above the box
    cv2.putText(frame, prediction_percent_str(top_n_predictions[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,emotion_to_color(top_n_predictions[0][0]), 2)
    cv2.putText(frame, prediction_percent_str(top_n_predictions[1]), (x , y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, prediction_percent_str(top_n_predictions[2]), (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
  # Display the resulting frame
  cv2.imshow('Emotion Detection', frame)


while True:
  try:
    predict_emotions()    
  except Exception as e:
    continue
  # Break the loop if 'q' is pressed
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()









