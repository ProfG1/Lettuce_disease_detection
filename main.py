import numpy as np
import cv2
from tensorflow.keras.models import load_model
import serial
import time
import os

# Initialize serial communication with Arduino
arduino = serial.Serial('COM3', 115200, timeout=1)  # Adjust COM port as necessary

# Load the pre-trained model
model = load_model('lettuce_disease_classifier.h5')
print("Model input shape:", model.input_shape)

# Function to capture image from webcam
def capture_image(sample_name):
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print(f"Error: Could not open webcam for {sample_name}.")
        return None

    ret, frame = cap.read()

    if ret:
        image_path = f'{sample_name}.jpg'
        cv2.imwrite(image_path, frame)
        print(f"Image saved as {image_path}")
    else:
        print(f"Failed to capture {sample_name}.")
        image_path = None

    cap.release()
    return image_path

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))  # Adjust based on model input
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Shape becomes (1, 150, 150, 3)
    return img

# Function to make predictions
def predict_disease(image_path):
    processed_img = preprocess_image(image_path)
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class, predictions

# Function to send data to Arduino
def send_to_arduino(data):
    arduino.write(data.encode())
    time.sleep(1)

# Main loop to capture and classify images
def main_loop():
    while True:
        key = input("Press 's' to start capturing images: ")

        if key == 's':
            print("Capturing sample A...")
            sample_a_path = capture_image('sample_a')

            if sample_a_path:
                predicted_class_a, predictions_a = predict_disease(sample_a_path)
                send_to_arduino("first_done")  # Notify Arduino that first capture is done
                print("Waiting for Arduino signal to capture second image...")

                # Wait for Arduino signal to take second picture
                while True:
                    arduino_signal = arduino.readline().decode().strip()
                    if arduino_signal == "capture_second":
                        print("Capturing sample B...")
                        sample_b_path = capture_image('sample_b')
                        break

                if sample_b_path:
                    predicted_class_b, predictions_b = predict_disease(sample_b_path)

                    # Send class labels of both samples to Arduino
                    send_to_arduino(f"A:{predicted_class_a[0]}, B:{predicted_class_b[0]}")
                    print(f"Sample A class: {predicted_class_a[0]}, Sample B class: {predicted_class_b[0]}")

            print("Waiting for next 's' key press...")

if __name__ == "__main__":
    main_loop()
