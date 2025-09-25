# 1: Import required libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import zipfile

# 2: Download data and set variables - FIXED PATH ISSUE
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

# Download dataset
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=False)
base_dir = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

# Extract manually to control the process
with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
    zip_ref.extractall(os.path.dirname(path_to_zip))

# Set correct paths
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Check if directories exist
print("Checking directories...")
print(f"Train directory exists: {os.path.exists(train_dir)}")
print(f"Validation directory exists: {os.path.exists(validation_dir)}")

if not os.path.exists(train_dir):
    # Try alternative path structure
    base_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets', 'cats_and_dogs_filtered')
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    print(f"Trying alternative path: {base_dir}")
    print(f"Train directory exists: {os.path.exists(train_dir)}")

# Parameters
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

# 3: Create image generators - FIXED
train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)

try:
    train_data_gen = train_image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='binary'
    )

    val_data_gen = validation_image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='binary'
    )
    
    print("Data generators created successfully!")
    
except Exception as e:
    print(f"Error creating data generators: {e}")
    print("Trying to find the correct directory structure...")
    
    # List all files and directories to debug
    for root, dirs, files in os.walk(os.path.dirname(path_to_zip)):
        level = root.replace(os.path.dirname(path_to_zip), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f'{subindent}{file}')
        if len(files) > 5:
            print(f'{subindent}... and {len(files) - 5} more files')
    
    exit()

# 4: Plot images function
def plotImages(images_arr, probabilities=False):
    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5, len(images_arr)*3))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for i, (img, ax) in enumerate(zip(images_arr, axes)):
        ax.imshow(img)
        ax.axis('off')
        if probabilities is not False:
            if isinstance(probabilities, (list, np.ndarray)) and i < len(probabilities):
                prob = probabilities[i]
            else:
                prob = probabilities
            if prob > 0.5:
                ax.set_title("%.2f" % (prob*100) + "% dog")
            else:
                ax.set_title("%.2f" % ((1-prob)*100) + "% cat")
    plt.tight_layout()
    plt.show()

# Plot sample training images
sample_training_images, _ = next(train_data_gen)
plotImages(sample_training_images[:5])

# 5: Add data augmentation
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

# 6: Create new train data generator with augmentation
train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)

# Plot augmented images
augmented_images = []
for i in range(5):
    batch = next(train_data_gen)
    augmented_images.append(batch[0][0])
plotImages(augmented_images)

# 7: Create the model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# 8: Train the model
print("Starting training...")
history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=val_data_gen.samples // batch_size,
    verbose=1
)

# 9: Visualize training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# 10: Create simple test data for demonstration
# Since the original test directory might not exist, we'll use some validation images as test
test_image_generator = ImageDataGenerator(rescale=1./255)

# Use a subset of validation images as test
test_data_gen = test_image_generator.flow_from_directory(
    batch_size=10,
    directory=validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary',
    shuffle=False
)

# Make predictions
test_data_gen.reset()
probabilities = model.predict(test_data_gen)

# Get test images
test_data_gen.reset()
test_images, _ = next(test_data_gen)

# Plot test images with predictions
plotImages(test_images[:5], probabilities[:5].flatten())

# 11: Evaluate results
val_accuracy = max(history.history['val_accuracy'])
final_accuracy = history.history['val_accuracy'][-1]
print(f"Best validation accuracy: {val_accuracy:.2%}")
print(f"Final validation accuracy: {final_accuracy:.2%}")

if final_accuracy >= 0.70:
    print("🎉 You passed the challenge with 70% or higher accuracy! Extra credit!")
elif final_accuracy >= 0.63:
    print("✅ You passed the challenge!")
else:
    print("❌ You haven't passed yet. Keep trying!")
    print("Try increasing the number of epochs or adjusting the model architecture.")