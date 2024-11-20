import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import cv2
import numpy as np
from keras import models

gender_model = models.load_model("models/gender_model.h5", compile=False)
ethnicity_model = models.load_model("models/ethnicity_model.h5", compile=False)
age_model = models.load_model("models/age_model.h5", compile=False)

text_font = cv2.FONT_HERSHEY_SIMPLEX
text_fontScale = 0.8
text_color = (0, 0, 255)
text_thickness = 1

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_attributes(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image at {image_path} could not be loaded.")
        return
    img = cv2.resize(img, (512, 512))
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        print("No faces detected.")
        return
    
    for (x, y, w, h) in faces:
        face = gray_img[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=(0, -1))
        
        gender_pred = gender_model.predict(face_input)
        ethnicity_pred = ethnicity_model.predict(face_input)
        age_pred = age_model.predict(face_input)

        gender = "Male" if gender_pred[0][0] < 0.5 else "Female"
        ethnicity = {
            0: "White", 1: "Black", 2: "Asiatic", 3: "Indian", 4: "Other"
        }.get(np.argmax(ethnicity_pred[0]), "Unknown")
        age = int(age_pred[0][0])

        cv2.putText(img, f"Gender: {gender}", (x, y - 50), text_font, text_fontScale, text_color, text_thickness)
        cv2.putText(img, f"Ethnicity: {ethnicity}", (x, y - 30), text_font, text_fontScale, text_color, text_thickness)
        cv2.putText(img, f"Age: {age}", (x, y - 10), text_font, text_fontScale, text_color, text_thickness)

    cv2.imshow('Facial Attribute Recognition', cv2.resize(img, (512, 512)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = 'imgs/merkel.jpg'
predict_attributes(image_path)
