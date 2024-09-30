import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras import load_model
import cv2
import numpy as np

# Load the trained model
model = load_model('../models/combined_model.h5')

# Start webcam feed
cap = cv2.VideoCapture(0)  # 0 means the default webcam

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    # Preprocess the frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized_frame = cv2.resize(gray_frame, (48, 48))  # Resize to 48x48 (or your model's input size)
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    input_frame = normalized_frame.reshape(1, 48, 48, 1)  # Reshape for the model

    # Make predictions (age, gender, ethnicity)
    age_pred, gender_pred, ethnicity_pred = model.predict(input_frame)

    # Post-process the predictions
    age = int(age_pred[0])
    gender = "Male" if gender_pred[0] < 0.5 else "Female"
    ethnicity_labels = ['White', 'Black', 'Asiatic', 'Indian', 'Other']
    ethnicity = ethnicity_labels[np.argmax(ethnicity_pred[0])]

    # Display the predictions on the frame
    cv2.putText(frame, f"Age: {age}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Gender: {gender}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Ethnicity: {ethnicity}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with predictions
    cv2.imshow('Real-Time Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()