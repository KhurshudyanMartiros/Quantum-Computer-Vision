import os
import numpy as np
import pandas as pd
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # Import tqdm for progress tracking

import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Enable mixed precision training
from tensorflow.keras.mixed_precision import global_policy, set_global_policy

policy = global_policy()
set_global_policy('mixed_float16')

n_epochs = 500  # Reduce for testing

SAVE_PATH = "quanvolution_3/"
PREPROCESS = True
np.random.seed(0)
tf.random.set_seed(0)

# Load and preprocess data
data = []
labels = []
classes = 43
cur_path = os.getcwd()

# Retrieving the images and their labels
for i in range(classes):
    path = os.path.join(cur_path, 'gtsrb-german-traffic-sign', 'train', str(i))
    images = os.listdir(path)

    for img_name in images:
        try:
            img_path = os.path.join(path, img_name)
            image = Image.open(img_path)
            image = image.resize((28, 28))
            image = np.array(image) / 255.0  # Normalize image
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

# Converting lists into numpy arrays
X_train = np.array(data)
y_train = np.array(labels)

# Apply to_categorical to the labels
y_train = keras.utils.to_categorical(y_train, num_classes=classes)

# Displaying the shape after the split
print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")

# Importing the test dataset
test_csv_path = os.path.join(cur_path, 'gtsrb-german-traffic-sign', 'Test.csv')
test_csv = pd.read_csv(test_csv_path)

y_test = test_csv["ClassId"].values
imgs = test_csv["Path"].values

data_test = []

# Retrieving the test images
for img in imgs:
    try:
        img_path = os.path.join(cur_path, 'gtsrb-german-traffic-sign', img)
        image = Image.open(img_path)
        image = image.resize((28, 28))
        image = np.array(image) / 255.0  # Normalize image
        data_test.append(image)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

X_test = np.array(data_test)

# Apply to_categorical to the test labels
y_test = keras.utils.to_categorical(y_test, num_classes=classes)

print(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

dev = qml.device("default.qubit", wires=4)

# Define the Grover's algorithm on two qubits
def grover_circuit(q0, q1):
    
    # Apply the Grover diffusion operator
    qml.Hadamard(wires=q0)
    qml.Hadamard(wires=q1)
    qml.PauliX(wires=q0)
    qml.PauliX(wires=q1)
    qml.Hadamard(wires=q1)
    qml.CNOT(wires=[q0, q1])
    #qml.ISWAP(wires=[q0,q1])
    qml.Hadamard(wires=q1)
    qml.PauliX(wires=q0)
    qml.PauliX(wires=q1)
    qml.Hadamard(wires=q0)
    qml.Hadamard(wires=q1)
    
@qml.qnode(dev, interface="autograd")
def circuit(phi):
    qml.RY(phi[0], wires=0)
    qml.RY(phi[1], wires=1)
    qml.RY(phi[2], wires=2)
    qml.RY(phi[3], wires=3)
    
    grover_circuit(0, 1)
        
    # Measurement producing 4 classical output values
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]

def quanv(image):
    # Convolves the input image with many applications of the same quantum circuit.
    out = np.zeros((14, 14, 4))

    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            # Process a squared 2x2 region of the image with a quantum circuit
            q_results = circuit(
                [
                    image[j, k, 0],
                    image[j, k + 1, 0],
                    image[j + 1, k, 0],
                    image[j + 1, k + 1, 0]
                ]
            )
            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            for c in range(4):
                out[j // 2, k // 2, c] = q_results[c]
    return out

def process_image(img):
    return quanv(img)

if __name__ == '__main__':
    if PREPROCESS:
        q_train_images = []
        print("Quantum pre-processing of train images:")
        for idx, img in enumerate(X_train):
            print("{}/{}        ".format(idx + 1, y_train.shape[0]), end="\r")
            q_train_images.append(quanv(img))
        q_train_images = np.asarray(q_train_images)

        q_test_images = []
        print("\nQuantum pre-processing of test images:")
        for idx, img in enumerate(X_test):
            print("{}/{}        ".format(idx + 1, y_test.shape[0]), end="\r")
            q_test_images.append(quanv(img))
        q_test_images = np.asarray(q_test_images)

        # Save pre-processed images
        np.save(SAVE_PATH + "q_train_images.npy", q_train_images)
        np.save(SAVE_PATH + "q_test_images.npy", q_test_images)


    # Load pre-processed images
    q_train_images = np.load(SAVE_PATH + "q_train_images.npy")
    q_test_images = np.load(SAVE_PATH + "q_test_images.npy")

    n_samples = 4
    n_channels = 4
    fig, axes = plt.subplots(1 + n_channels, n_samples, figsize=(10, 10))
    for k in range(n_samples):
        axes[0, 0].set_ylabel("Input")
        if k != 0:
            axes[0, k].yaxis.set_visible(False)
        axes[0, k].imshow(X_train[k, :, :, 0], cmap="gray")

        # Plot all output channels
        for c in range(n_channels):
            axes[c + 1, 0].set_ylabel("Output [ch. {}]".format(c))
            if k != 0:
                axes[c, k].yaxis.set_visible(False)
            axes[c + 1, k].imshow(q_train_images[k, :, :, c], cmap="gray")

    plt.tight_layout()
    plt.show()

    def MyModel():
        # Initializes and returns a custom Keras model which is ready to be trained.
        model = keras.models.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(classes, activation="softmax")
        ])
        model.compile(
            optimizer='adam',
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    q_model = MyModel()

    q_history = q_model.fit(
        q_train_images,
        y_train,
        validation_data=(q_test_images, y_test),
        batch_size=512,  # Increase batch size
        epochs=n_epochs,
        verbose=2,
    )

    # Plotting code
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

    ax1.plot(q_history.history["val_accuracy"], label="With quantum layer")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(q_history.history["val_loss"], label="With quantum layer")
    ax2.set_ylabel("Loss")
    ax2.set_ylim(top=2.5)
    ax2.set_xlabel("Epoch")
    ax2.legend()
    plt.tight_layout()
    plt.show()

