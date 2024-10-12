import cv2
import subprocess
import os

def capture_image():
    # Open a connection to the webcam (0 is usually the default webcam)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    ret, frame = cap.read()

    if ret:
        # Save the captured image to a file without displaying it
        image_path = 'captured_image.jpg'
        cv2.imwrite(image_path, frame)
        print(f"Image saved as {image_path}")
    else:
        print("Error: Failed to capture image.")
        image_path = None

    # Release the webcam
    cap.release()

    return image_path

def run_model_inference():
    # Run model_inference.py after capturing the image
    print("Running the model inference...")
    subprocess.run(['python', 'main.py'])

if __name__ == "__main__":
    # Capture image first
    image_path = capture_image()

    # If image was captured successfully, run model inference
    if image_path and os.path.exists(image_path):
        run_model_inference()
    else:
        print("No image captured. Exiting.")
