import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('lettuce_disease_classifier.h5')
print("Model input shape:", model.input_shape)  # Check the expected input shape

# Define a function to preprocess the image
def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    # Resize the image to the input size expected by the model
    img = cv2.resize(img, (150, 150))  # Adjust this size based on your model's input shape
    # Convert image to float32 and normalize
    img = img.astype('float32') / 255.0
    # Expand dimensions to match model input shape
    img = np.expand_dims(img, axis=0)  # Shape becomes (1, 150, 150, 3)
    return img

# Define a function to make predictions
def predict_disease(image_path):
    # Preprocess the image
    processed_img = preprocess_image(image_path)
    print("Processed image shape:", processed_img.shape)  # Check the processed image shape
    # Make a prediction
    predictions = model.predict(processed_img)
    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class, predictions

# Define a main function to run the detection
def main():
    image_path = 'captured_image.jpg'  # Use the image captured from the webcam
    predicted_class, predictions = predict_disease(image_path)

    # Mapping classes to disease names (adjust based on your model's training)
    class_labels = {0: 'Bacterial', 1: 'Fungal', 2: 'Viral', 3: 'Healthy', 4: 'Other', 5: 'Disease F'}  # Add as many diseases as your model has

    # Print the prediction results
    print(f"Predicted Class: {class_labels[predicted_class[0]]}")
    print(f"Prediction Probabilities: {predictions}")

    # Display the input image
    img = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {class_labels[predicted_class[0]]}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
