import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

from keras.models import load_model
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '../models/combined_model.h5')

model = load_model(model_path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (48, 48))
    normalized_frame = resized_frame / 255.0
    input_frame = normalized_frame.reshape(1, 48, 48, 1)

    age_pred, gender_pred, ethnicity_pred = model.predict(input_frame)

    age = int(age_pred[0])
    gender = "Male" if gender_pred[0] < 0.5 else "Female"
    ethnicity_labels = ['White', 'Black', 'Asiatic', 'Indian', 'Other']
    ethnicity = ethnicity_labels[np.argmax(ethnicity_pred[0])]

    cv2.putText(frame, f"Age: {age}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Gender: {gender}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Ethnicity: {ethnicity}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Real-Time Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()