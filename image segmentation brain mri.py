import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

train_path  = r"E:\train"
test_path = r"E:\test"
val_path  = r"E:\valid"

train_images = [image for image in os.listdir(train_path) if image[-3:] =='jpg' ]
test_images = [image for image in os.listdir(test_path) if image[-3:] =='jpg' ]
val_images = [image for image in os.listdir(val_path) if image[-3:] =='jpg' ]
len(train_images),len(test_images),len(val_images)

import glob
import os

train_annotations = glob.glob(os.path.join(train_path, '*.json'))
test_annotations = glob.glob(os.path.join(test_path, '*.json'))
val_annotations = glob.glob(os.path.join(val_path, '*.json'))

import json
train_annotations = json.load(open(train_annotations[0]))
test_annotations = json.load(open(test_annotations[0]))
val_annotations = json.load(open(val_annotations[0]))

len(train_annotations['annotations']),len(test_annotations['annotations']),len(val_annotations['annotations']),
#same length of anno n imgs

#visualize 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def visualize_random_images(n=5):
    # Check if images and annotations are loaded
    if 'images' not in train_annotations or 'annotations' not in train_annotations:
        print("No images or annotations found in train_annotations.")
        return

    # Select n random images
    indices = np.random.choice(len(train_annotations['images']), size=n, replace=False)
    images = [train_annotations['images'][i] for i in indices]
    annotations = [train_annotations['annotations'][i] for i in indices]

    # Visualize images and their annotations
    plt.figure(figsize=(12, 4 * n))
    for i, (img, ann) in enumerate(zip(images, annotations)):
        image_path = os.path.join(train_path, img['file_name'])
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Warning: Could not load image {img['file_name']}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the original image
        plt.subplot(n, 3, 3*i + 1)
        plt.imshow(image)
        plt.title("Original Image")

        # Draw segmentation on the image if it exists and is non-empty
        segmentation = ann.get('segmentation')
        if isinstance(segmentation, list) and len(segmentation) > 0:
            segmentation = np.array(segmentation[0], dtype=np.int32).reshape(-1, 2)
            cv2.polylines(image, [segmentation], isClosed=True, color=(0, 255, 0), thickness=2)

        plt.subplot(n, 3, 3*i + 2)
        plt.imshow(image)
        plt.title("Annotated Image")

        # Create and display mask if segmentation exists
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        if isinstance(segmentation, np.ndarray):
            cv2.fillPoly(mask, [segmentation], color=1)
        
        plt.subplot(n, 3, 3*i + 3)
        plt.imshow(mask, cmap='gray')
        plt.title("Mask")

    plt.tight_layout()
    plt.show()

visualize_random_images()



import os
import cv2
import numpy as np
from threading import Thread

def _train_masks():
    print('Generating train masks...')
    mask_dir = '/kaggle/working/train_masks/'
    os.makedirs(mask_dir, exist_ok=True)
    total_images = len(train_annotations['images'])
    for idx, (img, ann) in enumerate(zip(train_annotations['images'], train_annotations['annotations']), start=1):
        path = os.path.join(train_path, img['file_name'])
        mask_path = os.path.join(mask_dir, img['file_name'])
        
        # Load image in OpenCV
        image = cv2.imread(path)
        if image is None:
            print(f"Warning: Could not load image {img['file_name']}")
            continue
        
        # Create mask
        segmentation = np.array(ann['segmentation'][0], dtype=np.int32).reshape(-1, 2)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [segmentation], color=(255, 255, 255))
        
        # Save mask
        cv2.imwrite(mask_path, mask)
        print(f"Train mask {idx} / {total_images} generated")

def _test_masks():
    print('Generating test masks...')
    mask_dir = '/kaggle/working/test_masks/'
    os.makedirs(mask_dir, exist_ok=True)
    total_images = len(test_annotations['images'])
    for idx, (img, ann) in enumerate(zip(test_annotations['images'], test_annotations['annotations']), start=1):
        path = os.path.join(test_path, img['file_name'])
        mask_path = os.path.join(mask_dir, img['file_name'])
        
        # Load image in OpenCV
        image = cv2.imread(path)
        if image is None:
            print(f"Warning: Could not load image {img['file_name']}")
            continue

        # Create mask
        segmentation = np.array(ann['segmentation'][0], dtype=np.int32).reshape(-1, 2)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [segmentation], color=(255, 255, 255))
        
        # Save mask
        cv2.imwrite(mask_path, mask)
        print(f"Test mask {idx} / {total_images} generated")

def _val_masks():
    print('Generating validation masks...')
    mask_dir = '/kaggle/working/val_masks/'
    os.makedirs(mask_dir, exist_ok=True)
    total_images = len(val_annotations['images'])
    for idx, (img, ann) in enumerate(zip(val_annotations['images'], val_annotations['annotations']), start=1):
        path = os.path.join(val_path, img['file_name'])
        mask_path = os.path.join(mask_dir, img['file_name'])
        
        # Load image in OpenCV
        image = cv2.imread(path)
        if image is None:
            print(f"Warning: Could not load image {img['file_name']}")
            continue

        # Create mask
        segmentation = np.array(ann['segmentation'][0], dtype=np.int32).reshape(-1, 2)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [segmentation], color=(255, 255, 255))
        
        # Save mask
        cv2.imwrite(mask_path, mask)
        print(f"Validation mask {idx} / {total_images} generated")

def make_masks():
    threads = []
    threads.append(Thread(target=_train_masks))
    threads.append(Thread(target=_test_masks))
    threads.append(Thread(target=_val_masks))
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    print('Mask generation complete')

# Run the function
make_masks()

import os

def count_masks(mask_dir):
    # List all files in the directory
    files = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))]
    return len(files)

# Directories for the masks
train_mask_dir = '/kaggle/working/train_masks/'
test_mask_dir = '/kaggle/working/test_masks/'
val_mask_dir = '/kaggle/working/val_masks/'

# Count the number of mask files in each directory
train_mask_count = count_masks(train_mask_dir)
test_mask_count = count_masks(test_mask_dir)
val_mask_count = count_masks(val_mask_dir)

print(f"Number of train masks: {train_mask_count}")
print(f"Number of test masks: {test_mask_count}")
print(f"Number of validation masks: {val_mask_count}")

import matplotlib.pyplot as plt
import cv2
import os
import random

def visualize_masks(mask_dir, n=5):
    # List all mask files in the directory
    mask_files = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))]
    
    # Randomly select n masks to display
    selected_masks = random.sample(mask_files, min(n, len(mask_files)))
    
    plt.figure(figsize=(12, 4 * len(selected_masks)))
    for i, mask_file in enumerate(selected_masks, start=1):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        plt.subplot(len(selected_masks), 1, i)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Mask: {mask_file}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Directories for train, test, and validation masks
train_mask_dir = '/kaggle/working/train_masks/'
test_mask_dir = '/kaggle/working/test_masks/'
val_mask_dir = '/kaggle/working/val_masks/'

# Visualize masks from each set
print("Visualizing train masks...")
visualize_masks(train_mask_dir)

print("Visualizing test masks...")
visualize_masks(test_mask_dir)

print("Visualizing validation masks...")
visualize_masks(val_mask_dir)

import cv2
import numpy as np
import os

# Corrected load_data function for train and validation data
def load_data():
    target_size = (128, 128)
    train_mask_dir = '/kaggle/working/train_masks/'
    val_mask_dir = '/kaggle/working/val_masks/'
    
    # Load train images and masks
    X_train = [cv2.resize(cv2.imread(os.path.join(train_path, image['file_name'])), target_size) 
               for image in train_annotations['images']]
    y_train = [cv2.resize(cv2.imread(os.path.join(train_mask_dir, image['file_name']), cv2.IMREAD_GRAYSCALE), target_size) 
               for image in train_annotations['images']]
    
    X_train = np.array(X_train)
    y_train = np.expand_dims(np.array(y_train), axis=-1)
    
    # Normalize images and binary threshold masks
    X_train = X_train.astype('float32') / 255.0
    y_train = y_train.astype('float32') / 255.0
    y_train = (y_train > 0.5).astype(np.float32)
    
    # Load validation images and masks
    X_val = [cv2.resize(cv2.imread(os.path.join(val_path, image['file_name'])), target_size) 
             for image in val_annotations['images']]
    y_val = [cv2.resize(cv2.imread(os.path.join(val_mask_dir, image['file_name']), cv2.IMREAD_GRAYSCALE), target_size) 
             for image in val_annotations['images']]
    
    X_val = np.array(X_val)
    y_val = np.expand_dims(np.array(y_val), axis=-1)
    
    # Normalize images and binary threshold masks
    X_val = X_val.astype('float32') / 255.0
    y_val = y_val.astype('float32') / 255.0
    y_val = (y_val > 0.5).astype(np.float32)
    
    return X_train, y_train, X_val, y_val

# Corrected load_test_data function for test data
def load_test_data():
    target_size = (128, 128)
    test_mask_dir = '/kaggle/working/test_masks/'
    
    # Load test images and masks
    X_test = [cv2.resize(cv2.imread(os.path.join(test_path, image['file_name'])), target_size) 
              for image in test_annotations['images']]
    y_test = [cv2.resize(cv2.imread(os.path.join(test_mask_dir, image['file_name']), cv2.IMREAD_GRAYSCALE), target_size) 
              for image in test_annotations['images']]
    
    X_test = np.array(X_test)
    y_test = np.expand_dims(np.array(y_test), axis=-1)
    
    # Normalize images and binary threshold masks
    X_test = X_test.astype('float32') / 255.0
    y_test = y_test.astype('float32') / 255.0
    y_test = (y_test > 0.5).astype(np.float32)
    
    return X_test, y_test

# Run the function to load the data
X_train, y_train, X_val, y_val = load_data()
X_test, y_test = load_test_data()

import matplotlib.pyplot as plt

def check_data(X, y, dataset_name="Dataset"):
    print(f"{dataset_name} - Images shape: {X.shape}")
    print(f"{dataset_name} - Masks shape: {y.shape}")
    print(f"{dataset_name} - Data type of images: {X.dtype}")
    print(f"{dataset_name} - Data type of masks: {y.dtype}")
    print(f"{dataset_name} - Unique values in masks: {np.unique(y)}\n")

def visualize_samples(X, y, n=3, dataset_name="Dataset"):
    plt.figure(figsize=(10, 4 * n))
    for i in range(n):
        # Display image
        plt.subplot(n, 2, 2 * i + 1)
        plt.imshow(X[i])
        plt.title(f"{dataset_name} Image {i + 1}")
        plt.axis('off')
        
        # Display corresponding mask
        plt.subplot(n, 2, 2 * i + 2)
        plt.imshow(y[i].squeeze(), cmap='gray')
        plt.title(f"{dataset_name} Mask {i + 1}")
        plt.axis('off')
    plt.show()

# Check and print data information
check_data(X_train, y_train, "Train")
check_data(X_val, y_val, "Validation")
check_data(X_test, y_test, "Test")

# Visualize a few samples from each set
print("Visualizing train samples:")
visualize_samples(X_train, y_train, n=3, dataset_name="Train")

print("Visualizing validation samples:")
visualize_samples(X_val, y_val, n=3, dataset_name="Validation")

print("Visualizing test samples:")
visualize_samples(X_test, y_test, n=3, dataset_name="Test")

#okk


#modelling 

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout #type: ignore
from tensorflow.keras.models import Model #type: ignore

def unet_model(input_size=(128, 128, 3)):
    inputs = Input(input_size)
    
    # Encoder
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # Bottleneck
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = Dropout(0.3)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    
    # Decoder
    u5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    
    u6 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(u6)
    c6 = Dropout(0.1)(c6)
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(u7)
    c7 = Dropout(0.1)(c7)
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(c7)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Create the model
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary to understand its structure
model.summary()

# Set training parameters
epochs = 10          # Adjust based on resources and performance needs
batch_size = 8       # Small batch size to avoid memory issues

# Train the model with your data
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Prediction function
def predict(image_path, model, threshold=0.5):
    # Read and preprocess the image
    image = cv2.imread(image_path)
    print("Original Image Shape:", image.shape)
    
    # Resize the image to match model's expected sizing
    resized_image = cv2.resize(image, (128, 128))
    print("Resized Image Shape:", resized_image.shape)
    
    # Expand dimensions to match the batch size used by the model
    input_image = np.expand_dims(resized_image, axis=0)
    
    # Preprocess input (normalize to [0, 1] range)
    input_image = input_image.astype('float32') / 255.0
    
    # Perform prediction
    pred_mask = model.predict(input_image)
    
    # Apply threshold to prediction mask
    pred_mask = (pred_mask >= threshold).astype(np.uint8)  # Set values >= threshold to 1, < threshold to 0
    
    # Squeeze to remove extra dimension
    pred_mask = np.squeeze(pred_mask, axis=0)
    print("Predicted Mask Shape:", pred_mask.shape)
    
    return pred_mask

# Function to display random test images with true and predicted masks
def test_random_images(model, n=5):
    test_mask_dir = '/kaggle/working/test_masks/'
    indices = np.random.randint(0, len(test_annotations['images']), size=n)
    images = [test_annotations['images'][i] for i in indices]
    
    plt.figure(figsize=(12, 4 * n))
    for i, img_data in enumerate(images):
        image_path = os.path.join(test_path, img_data['file_name'])
        true_mask_path = os.path.join(test_mask_dir, img_data['file_name'])
        
        # Load input image and true mask
        input_image = cv2.imread(image_path)
        true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Get predicted mask
        pred_mask = predict(image_path, model)
        
        # Display input image
        plt.subplot(n, 3, 3 * i + 1)
        plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        plt.title("Input Image")
        plt.axis('off')
        
        # Display true mask
        plt.subplot(n, 3, 3 * i + 2)
        plt.imshow(true_mask, cmap='gray')
        plt.title("True Mask")
        plt.axis('off')
        
        # Display predicted mask
        plt.subplot(n, 3, 3 * i + 3)
        plt.imshow(pred_mask, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Run the visualization function with the trained model
test_random_images(model, n=5)

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def single_prediction(image_path, mask_path, model, threshold=0.5):
    # Load and display the input image
    input_image = cv2.imread(image_path)
    true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Predict mask using the model
    pred_mask = predict(image_path, model, threshold=threshold)
    
    # Plot the input image, true mask, and predicted mask side-by-side
    plt.figure(figsize=(12, 4))

    # Display input image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")
    plt.axis('off')

    # Display true mask
    plt.subplot(1, 3, 2)
    plt.imshow(true_mask, cmap='gray')
    plt.title("True Mask")
    plt.axis('off')

    # Display predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.show()

# Example usage of the function with a single test image
image_path = os.path.join(test_path, test_annotations['images'][0][r"E:\test\215_jpg.rf.8831964c43bb52bfce62d3a6d060f8e8.jpg"])
mask_path = os.path.join('/kaggle/working/test_masks/', test_annotations['images'][0][r"E:\test\215_jpg.rf.8831964c43bb52bfce62d3a6d060f8e8.jpg"])



# Manually specify the filename you want to use
image_file_name = r"E:\test\215_jpg.rf.8831964c43bb52bfce62d3a6d060f8e8.jpg"  # Replace with your desired filename

# Construct the full paths for the image and corresponding mask
image_path = os.path.join(test_path, image_file_name)
mask_path = os.path.join('/kaggle/working/test_masks/', image_file_name)

# Run single prediction
single_prediction(image_path, mask_path, model)


import matplotlib.pyplot as plt

# Plot training & validation accuracy and loss values
def plot_training_history(history):
    # Plot loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_training_history(history)

# Path to the local image
image_path = r #type: ignore

# Run the prediction function on the local image
predicted_mask = predict(image_path, model)

# Display the input image and predicted mask
import matplotlib.pyplot as plt

# Read and display the original image
input_image = cv2.imread(image_path)
plt.figure(figsize=(10, 5))

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title("Input Image")
plt.axis('off')

# Show the predicted mask
plt.subplot(1, 2, 2)
plt.imshow(predicted_mask, cmap='gray')
plt.title("Predicted Mask")
plt.axis('off')

plt.show()
