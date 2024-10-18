import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models\combined_model.h5")
image_path = 'imgs\oneal.jpg'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_attributes(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Image at {image_path} could not be loaded.")
        return
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        print("No faces detected.")
        return
    
    for (x, y, w, h) in faces:
        face = gray_img[y : y + h, x : x + w]
        
        face_resized = cv2.resize(face, (48, 48))
        
        face_normalized = face_resized / 255.0
        
        face_input = np.expand_dims(face_normalized, axis=(0, -1))
        
        gender_pred, ethnicity_pred, age_pred = model.predict(face_input)
        
        gender = "Male" if gender_pred[0][0] < 0.5 else "Female"
        match np.argmax(ethnicity_pred[0]) - 1:
            case 0:
                ethnicity = "White"
            case 1:
                ethnicity = "Black"
            case 2:
                ethnicity = "Asiatic"
            case 3:
                ethnicity = "Indian"
            case 4:
                ethnicity = "Other"
        age = age_pred[0][0]//1000000 + 6
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, f"Gender: {gender}", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Ethnicity: {ethnicity}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Age: {int(age)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Facial Attribute Recognition', img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

predict_attributes(image_path)