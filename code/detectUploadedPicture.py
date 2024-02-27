import cv2
from tkinter import filedialog
from tkinter import Tk
from PIL import Image, ImageTk
import numpy as np
import dlib
from keras.models import load_model

def prediction_percent_str(prediction):
    return str(prediction[0]) + ": " + "{:.2f}".format(prediction[1] * 100) + "%"

def emotion_to_color(emotion):
    color_mapping = {
        'angry': (0, 0, 255),    # Red
        'happy': (0, 255, 0),    # Green
        'sad': (255, 0, 0),      # Blue
        'surprised': (255, 255, 0),  # Yellow
        'neutral': (128, 128, 128),  # Gray
        'disgusted': (255, 255, 255), # White
        'fearful': (255, 255, 255), # White
    }
    return color_mapping.get(emotion, (255, 255, 255))

class_to_label = {0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprised'}

emotion_model = load_model('../saved_models/model_1.h5')
IMG_SIZE = 96
dlib_detector = dlib.get_frontal_face_detector()

def analyze_image(file_path):
    image = cv2.imread(file_path)
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = dlib_detector(frame)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_roi = frame[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
        face_roi = np.reshape(face_roi, (IMG_SIZE, IMG_SIZE, 1))
        face_roi = np.array(face_roi) / 255.0
        prediction = emotion_model.predict(np.array([face_roi]))
        top_n_indices = np.argsort(prediction[0])[::-1][:3]
        top_n_predictions = [(class_to_label[i], prediction[0][i]) for i in top_n_indices]
        predicted_emotion = class_to_label[np.argmax(prediction)]
        cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_to_color(top_n_predictions[0][0]), 2)
        cv2.putText(frame, prediction_percent_str(top_n_predictions[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,emotion_to_color(top_n_predictions[0][0]), 2)
        cv2.putText(frame, prediction_percent_str(top_n_predictions[1]), (x , y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, prediction_percent_str(top_n_predictions[2]), (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

def choose_image():
    root = Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])

    if file_path:
        analyzed_image = analyze_image(file_path)
        analyzed_image.show()  # Display the analyzed image using the default image viewer
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if save_path:
            analyzed_image.save(save_path)

if __name__ == "__main__":
    choose_image()
