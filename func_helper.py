# Necessary imports
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import datetime

class_names = ["Lavender", "Rose", "Lilly", "Sunflower"]


# Walking through the directory and finding the length of each class
def walk_through_dir(dir_path):
    """
    Takes the directory path, and find the number of directories, images inside it.
    Useful for quickly checking how many images there are.

    Args:
        dir_path (str): The desired directory to check from.
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories, {len(filenames)} images in '{dirpath}'")


# Displaying a random image
def view_random_image(dir_path, class_name=None):
    """
    Displays a random image from the desired path (e.g. to check the quality of the images).
    By default, `class_name` is set to None. If None, then a random class will be chosen
    from predefined list of classes `class_names`.

    Args:
        dir_path (str): The desired path, to display a random imagr from.
        class_name (str): The desired class to visualize.
    """
    # Get the class filepath
    if class_name is None:
        class_name = random.choice(class_names)
        full_dirpath = dir_path + "/" + class_name
    else:
        full_dirpath = dir_path + "/" + class_name

    # Get the random image
    rand_img = random.choice(os.listdir(full_dirpath))
    img_path = full_dirpath + "/" + rand_img

    # Visualize the random image
    img = plt.imread(img_path)
    plt.axis(False)
    plt.title(class_name)
    plt.imshow(img)


# Creating TensorBoard callbacks
def create_tensorboard_callback(dir_path, experiment_name):
    """
    Takes a directory path `dir_path`, the name of the experiment `experiment_name`,
    and returns an instance of TensorBoard Callback, ready to be used with TensorBoard.

    Args:
        dir_path (str): The desired directory, to save the TensorBoard Callback instance to.
        experiment_name (str): The desired name of the experiment, to differentiate between various experiments.

    Returns:
        tensorboard_callback instance.
    """
    # Defining the desired logging directory, and creating a callback instance.
    log_dir = dir_path + "/" + experiment_name + datetime.datetime.now().strftime("%H-%M-%S-%d-%m-%Y")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    # Some communication ;)
    print(f"Creating TensorBoard Callback at: {log_dir}")
    return tensorboard_callback


# Plotting losses & accuracies
def plot_loss_acc(history):
    """
    Takes a history instance, and plots 2 seperate plots of `loss` vs. `val_loss`,
    and `accuracy` vs. `val_accuracy` - on the time domain of epochs.

    Args:
        history: The instance of model's training history.
    """
    # Define loss/accuracy etc...
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    epochs = range(len(history.history["loss"]))

    # Actual plotting :D
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="validation_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.figure()
    plt.plot(epochs, accuracy, label="training_acc")
    plt.plot(epochs, val_accuracy, label="validation_acc")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


# Creating model
def create_model(model_url, num_classes):
    """
    Takes model's URL from Kaggle and creates an uncompiled Keras Sequential model with it.
    Returns a categorical feature extractor model.

    Args:
        model_url (str): A weblink of the desired model to utilize.

    Returns:
        An uncompiled Keras Sequential model with `model_url` as feature extraction
        layer, and Dense layer with `NUM_CLASSES` output neurons.
    """

    model = tf.keras.Sequential([
        hub.KerasLayer(model_url,
                       trainable=False,
                       input_shape=(224, 224, 3),
                       name="feature_extractor_layer"),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output_layer")
    ])
    return model


# Preprocessing custom images
def preprocess_image(dir_path, img_shape=224, scale=True):
    """
    Reads in an image from a filepath `dir_path`, turns it into a Tensor, reshapes it to `img_shape`,
    and scales the image if desired.

    Args:
        dir_path (str): The path leading to the image.
        img_shape (int): The desired shape of the image (width & height). By default, set to 224.
        scale (bool): Should the image be scaled to values between 0-1.

    Returns:
        A preprocessed version of an input image - turned into a Tensor, reshaped,
        and (if desired) - rescaled.
    """
    img = tf.io.read_file(filename=dir_path)  # read in
    img = tf.image.decode_jpeg(img)  # decode JPEG (!)
    img = tf.image.resize(img, size=[img_shape, img_shape])  # resizing
    if scale:  # optional scaling
        return img / 255.
    return img


# Predicting & plotting custom IMG
def pred_and_plot(model, dir_path, class_names=class_names):
    """
    Takes a model, an image filepath and makes a prediction with the given model.
    Plots the image with the predicted class as the title.

    Args:
        model: An instance of a model to make a prediction with.
        dir_path (str): The filepath of a desired image to make a prediction on.
        class_names (list): A list consisting of available class names.
    """
    # Preprocess IMG -> Make a prediction -> Get the predicted class and model's confidence
    preprocessed_img = preprocess_image(dir_path=dir_path)
    preds = model.predict(tf.expand_dims(preprocessed_img, axis=0))
    pred_index = tf.argmax(preds.reshape(-1))
    confidence = preds.reshape(-1)[pred_index]
    pred_class = class_names[pred_index]

    # Plot the img and the prediction
    plt.imshow(preprocessed_img)
    plt.title(f"Prediction: {pred_class}, Confidence: {confidence}")
    plt.axis(False)
    plt.show()