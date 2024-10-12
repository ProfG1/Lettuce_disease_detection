# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset path (update with your correct path)
dataset_folder_path = r'C:\Users\acer\Desktop\LettuceBegin\dataset\Lettuce disease datasets'

# Preprocessing - Image Augmentation
data_gen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.2  # Split into training and validation sets
)

# Training data generator
train_generator = data_gen.flow_from_directory(
    dataset_folder_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Use this subset for training
)

# Validation data generator
validation_generator = data_gen.flow_from_directory(
    dataset_folder_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Use this subset for validation
)

# Get class labels
class_indices = train_generator.class_indices
print("Class indices: ", class_indices)

# CNN Model Definition
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_indices), activation='softmax')  # Output layer with softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,  # Training data
    epochs=10,  # Number of epochs to train
    validation_data=validation_generator  # Validation data
)

# Save the trained model
model.save('lettuce_disease_classifier.h5')

# Evaluate the model on the validation data
val_loss, val_acc = model.evaluate(validation_generator)
print(f'Validation accuracy: {val_acc:.2f}')
