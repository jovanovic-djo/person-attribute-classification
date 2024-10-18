import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("models\combined_model.h5")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]

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

        # Draw rectangle around the face and put labels
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"Gender: {gender}", (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Ethnicity: {ethnicity}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Age: {int(age)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Facial Attribute Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
