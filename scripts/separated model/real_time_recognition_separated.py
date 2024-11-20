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
text_fontScale = 0.6
text_color = (255, 255, 255)
text_thickness = 2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_attributes(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face = gray_frame[y:y + h, x:x + w]
        
        face_resized = cv2.resize(face, (48, 48))
        
        face_normalized = face_resized / 255.0
        
        face_input = np.expand_dims(face_normalized, axis=(0, -1))
        
        gender_pred = gender_model.predict(face_input)
        ethnicity_pred = ethnicity_model.predict(face_input)
        age_pred = age_model.predict(face_input)
        
        gender = "Male" if gender_pred[0][0] < 0.5 else "Female"
        ethnicity_index = np.argmax(ethnicity_pred[0])
        ethnicity = {
            0: "White",
            1: "Black",
            2: "Asiatic",
            3: "Indian",
            4: "Other"
        }.get(ethnicity_index, "Unknown")
        age = int(age_pred[0][0])
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)       
        cv2.putText(frame, f"Gender: {gender}", (x, y - 50), text_font, text_fontScale, text_color, text_thickness)        
        cv2.putText(frame, f"Ethnicity: {ethnicity}", (x, y - 30), text_font, text_fontScale, text_color, text_thickness)
        cv2.putText(frame, f"Age: {age}", (x, y - 10), text_font, text_fontScale, text_color, text_thickness)
    
    return frame


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    annotated_frame = predict_attributes(frame)
    
    cv2.imshow('Facial Attribute Recognition', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
