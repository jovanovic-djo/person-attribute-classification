import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model(r"C:\\Users\\gatz0\\Desktop\\Projects\\person-attribute-classification\\models\\combined_model.h5")

# Load pre-trained face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from video capture
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert frame to grayscale (since the model expects a 48x48 grayscale input)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest (the face) from the frame
        face = gray_frame[y:y+h, x:x+w]

        # Resize to 48x48 (as expected by your model)
        face_resized = cv2.resize(face, (48, 48))

        # Normalize the pixel values (0-255) to the range 0-1
        face_normalized = face_resized / 255.0

        # Reshape to match the input shape your model expects (1, 48, 48, 1)
        face_input = np.expand_dims(face_normalized, axis=(0, -1))

        # Call the model to predict gender, ethnicity, and age
        gender_pred, ethnicity_pred, age_pred = model.predict(face_input)

        # Decode predictions
        gender = "Male" if gender_pred[0][0] < 0.5 else "Female"
        ethnicity = np.argmax(ethnicity_pred[0])
        age = age_pred[0][0]

        # Draw rectangle around the face and put labels
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"Gender: {gender}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Ethnicity: {ethnicity}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Age: {int(age)}", (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the frame with the detections and predictions
    cv2.imshow('Real-time Facial Attribute Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
