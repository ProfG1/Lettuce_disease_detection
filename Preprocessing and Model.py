import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Set paths for the dataset categories
dataset_folder_path = r'C:\Users\acer\Desktop\LettuceBegin\dataset\Lettuce disease datasets'
batch_size = 32
image_size = (150, 150)

# Preprocessing - Image Data Augmentation
data_gen = ImageDataGenerator(
    rescale=1./255,           # Normalization
    validation_split=0.2      # Split the data into training and validation sets
)

# Create training and validation data generators
train_generator = data_gen.flow_from_directory(
    dataset_folder_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Multi-class classification
    subset='training'
)

validation_generator = data_gen.flow_from_directory(
    dataset_folder_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Get class labels mapping for reference
class_indices = train_generator.class_indices
print("Class indices: ", class_indices)
