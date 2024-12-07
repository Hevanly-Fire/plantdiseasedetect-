

import os
import zipfile

def extract_zip(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def total_files(folder_path):
    num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    return num_files


# In[16]:


# Define the path of my dataset zip file
dataset_zip_path = "dataset.zip"

# Temporary directory to extract zip file
temp_extract_dir = "temp_extract"

# Extract the zip file
extract_zip(dataset_zip_path, temp_extract_dir)

# Define paths to the extracted dataset folders
train_files_healthy = os.path.join(temp_extract_dir, "Train/Train/Healthy")
train_files_powdery = os.path.join(temp_extract_dir, "Train/Train/Powdery")
train_files_rust = os.path.join(temp_extract_dir, "Train/Train/Rust")

test_files_healthy = os.path.join(temp_extract_dir, "Test/Test/Healthy")
test_files_powdery = os.path.join(temp_extract_dir, "Test/Test/Powdery")
test_files_rust = os.path.join(temp_extract_dir, "Test/Test/Rust")

valid_files_healthy = os.path.join(temp_extract_dir, "Validation/Validation/Healthy")
valid_files_powdery = os.path.join(temp_extract_dir, "Validation/Validation/Powdery")
valid_files_rust = os.path.join(temp_extract_dir, "Validation/Validation/Rust")


# In[17]:


# Print the number of files in each dataset folder
print("Number of healthy leaf images in training set:", total_files(train_files_healthy))
print("Number of powdery leaf images in training set:", total_files(train_files_powdery))
print("Number of rusty leaf images in training set:", total_files(train_files_rust))
print("========================================================")
print("Number of healthy leaf images in test set:", total_files(test_files_healthy))
print("Number of powdery leaf images in test set:", total_files(test_files_powdery))
print("Number of rusty leaf images in test set:", total_files(test_files_rust))
print("========================================================")
print("Number of healthy leaf images in validation set:", total_files(valid_files_healthy))
print("Number of powdery leaf images in validation set:", total_files(valid_files_powdery))
print("Number of rusty leaf images in validation set:", total_files(valid_files_rust))



# In[18]:


from PIL import Image
import IPython.display as display
import io

# Path to the zip file
zip_file_path = 'dataset.zip'
# Path to the image within the zip file
image_path_inside_zip = 'Train/Train/Healthy/8ce77048e12f3dd4.jpg'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Read the image file as bytes
    with zip_ref.open(image_path_inside_zip) as image_file:
        # Open the image using PIL
        img = Image.open(io.BytesIO(image_file.read()))
        # Display the image
        display.display(img)


# In[19]:


zip_file_path = 'dataset.zip'
image_path_inside_zip = 'Train/Train/Rust/80f09587dfc7988e.jpg'

# Open the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Read the image file as bytes
    with zip_ref.open(image_path_inside_zip) as image_file:
        # Open the image using PIL
        img = Image.open(io.BytesIO(image_file.read()))
        # Display the image
        display.display(img)


# In[36]:


import os
import shutil
import tempfile
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input


# In[37]:


# Define data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[38]:


def extract_and_load_images(zip_file_path, target_size, batch_size, class_mode):
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)  # Extract the contents of the zip file to the temporary directory

        train_dir = os.path.join(temp_dir, 'Test', 'Test')
        validation_dir = os.path.join(temp_dir, 'Validation', 'Validation')

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode
        )

        validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode
        )

    return train_generator, validation_generator

# Parameters
zip_file_path = 'C:\\Users\\shiva\\PycharmProjects\\arpii\\arpii\\dataset.zip'  # Update the zip file path
target_size = (225, 225)
batch_size = 16
class_mode = 'categorical'

# Load images from zip file
train_generator, validation_generator = extract_and_load_images(zip_file_path, target_size, batch_size, class_mode)


# In[40]:


# Define the model
model = Sequential([
    Input(shape=(225, 225, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])


# In[41]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[42]:


history = model.fit(train_generator,
                    batch_size=16,
                    epochs=5,
                    validation_data=validation_generator,
                    validation_batch_size=16
                    )


# In[51]:


from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

import seaborn as sns
sns.set_theme()
sns.set_context("poster")

figure(figsize=(25, 25), dpi=100)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[44]:


model.save("model.h5")


# In[46]:


from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import zipfile
import io

def preprocess_image_from_zip(image_path, target_size=(225, 225)):
    with open(image_path, 'rb') as f:
        img = load_img(io.BytesIO(f.read()), target_size=target_size)
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    return x
def startimg(path):
    x = preprocess_image_from_zip(path)
    predictions = model.predict(x)
    predictions[0]
    labels = train_generator.class_indices
    labels = {v: k for k, v in labels.items()}
    labels
    predicted_label = labels[np.argmax(predictions)]
    return predicted_label

