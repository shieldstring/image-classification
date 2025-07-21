import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- 1. Load and Preprocess CIFAR-10 Dataset ---
print("--- Loading and Preprocessing CIFAR-10 Dataset ---")

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
# CIFAR-10 has 10 classes
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Number of classes: {num_classes}")
print("Dataset loaded and preprocessed successfully.")

# Define class names for better readability in plots
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# --- 2. Define a function to create CNN models ---
def create_cnn_model(input_shape, num_classes, architecture_type='simple_cnn', learning_rate=0.001, optimizer_type='adam'):
    """
    Creates a Convolutional Neural Network (CNN) model with specified architecture and optimizer.

    Args:
        input_shape (tuple): Shape of the input images (e.g., (32, 32, 3)).
        num_classes (int): Number of output classes.
        architecture_type (str): Defines the CNN architecture ('simple_cnn', 'deeper_cnn').
        learning_rate (float): Learning rate for the optimizer.
        optimizer_type (str): Type of optimizer ('adam', 'sgd').

    Returns:
        tf.keras.Model: Compiled Keras CNN model.
    """
    model = Sequential()

    if architecture_type == 'simple_cnn':
        # Simple CNN architecture
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25)) # Regularization to prevent overfitting

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten()) # Flatten the 3D output to 1D for dense layers
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax')) # Output layer with softmax for multi-class

    elif architecture_type == 'deeper_cnn':
        # Deeper CNN architecture with BatchNormalization
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
        model.add(BatchNormalization()) # Helps stabilize and accelerate training
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

    else:
        raise ValueError("Invalid architecture_type. Choose 'simple_cnn' or 'deeper_cnn'.")

    # Choose optimizer
    if optimizer_type == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True) # SGD with Nesterov momentum
    else:
        raise ValueError("Invalid optimizer_type. Choose 'adam' or 'sgd'.")

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- 3. Hyper-parameter Optimization / Experimentation ---
print("\n--- Starting Hyper-parameter Optimization Experiments ---")

experiments = {
    "Simple CNN - Adam (LR=0.001)": {
        "architecture_type": "simple_cnn",
        "learning_rate": 0.001,
        "optimizer_type": "adam",
        "epochs": 20,
        "batch_size": 64
    },
    "Deeper CNN - Adam (LR=0.001)": {
        "architecture_type": "deeper_cnn",
        "learning_rate": 0.001,
        "optimizer_type": "adam",
        "epochs": 20,
        "batch_size": 64
    },
    "Simple CNN - SGD (LR=0.01)": {
        "architecture_type": "simple_cnn",
        "learning_rate": 0.01,
        "optimizer_type": "sgd",
        "epochs": 20,
        "batch_size": 64
    }
}

results = {}

for exp_name, params in experiments.items():
    print(f"\n--- Running Experiment: {exp_name} ---")
    model = create_cnn_model(
        input_shape=X_train.shape[1:],
        num_classes=num_classes,
        architecture_type=params["architecture_type"],
        learning_rate=params["learning_rate"],
        optimizer_type=params["optimizer_type"]
    )
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        validation_split=0.1, # Use a validation split from training data
        verbose=1 # Show training progress
    )

    # Evaluate the model on the test data
    print(f"\n--- Evaluating {exp_name} on Test Set ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Generate classification report and confusion matrix
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    results[exp_name] = {
        "model": model,
        "history": history,
        "test_loss": loss,
        "test_accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm
    }

# --- 4. Visualization of Results ---
print("\n--- Generating Visualizations ---")

for exp_name, res in results.items():
    history = res["history"]
    cm = res["confusion_matrix"]

    # Plot Training & Validation Accuracy and Loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{exp_name} - Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{exp_name} - Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{exp_name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

print("\n--- All Experiments Completed ---")