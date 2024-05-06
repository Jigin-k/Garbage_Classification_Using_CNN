import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load your pre-trained CNN model (replace 'your_model.h5' with your model file)
model = keras.models.load_model('C:/@@@/deep learning/cnn_garbage/garbage_classification.h5')

# Define class labels (replace with your own labels)
class_labels = ['cardboard','glass','metal','paper','plastic','trash']

def detect_and_classify_garbage_from_webcam():
    cap = cv2.VideoCapture(0)  # Use the default camera (0) or specify another camera if needed

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Preprocess the frame
        resized_frame = cv2.resize(frame, (150,150))  # Resize frame to match your model input size
        preprocessed_frame = resized_frame / 255.0  # Normalize pixel values (assuming your model expects this)

        # Make predictions using the model
        predictions = model.predict(np.expand_dims(preprocessed_frame, axis=0))

        # Get the predicted class
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_index]

        # Display the classification result on the frame
        cv2.putText(frame, f'Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with classification result
        cv2.imshow('Garbage Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_classify_garbage_from_webcam()
