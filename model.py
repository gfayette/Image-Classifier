import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
from os import listdir
from os.path import isfile, join


class Model:
    def __init__(self, c_path):
        # Training and validation data sets
        self.train_ds = None
        self.val_ds = None

        # Data parameters
        self.img_height = 180
        self.img_width = 180
        self.batch_size = 32

        # Classifications
        self.class_names = None
        self.num_classes = None

        # Model
        self.model = None
        self.num_epochs = None
        self.history = None

        # Save and load
        self.checkpoint_path = c_path
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        # string for window
        self.prediction_string = None

    def download(self, dataset_url, folder_name):
        """Only used when the data set needs to be downloaded

        """
        data_dir = tf.keras.utils.get_file(folder_name, origin=dataset_url, untar=True)
        data_dir = pathlib.Path(data_dir)
        print("\nData downloaded to: " + str(data_dir))
        return data_dir

    def get_and_split_data_set(self, data_dir):
        """For data that is not split into training and validation

        """
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.class_names = self.train_ds.class_names
        self.num_classes = len(self.class_names)
        self.configure_data_sets()


    def get_data_sets(self, train_dir, val_dir):
        """For data that is already split into training and validation

        """
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            val_dir,
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.class_names = self.train_ds.class_names
        self.num_classes = len(self.class_names)
        self.configure_data_sets()

    def configure_data_sets(self):
        """Configure dataset for performance

        """
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def create_and_compile_model(self):
        """Create a model with data augmentation and compile it

        """
        # Data augmentation
        data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal",
                                                             input_shape=(self.img_height,
                                                                          self.img_width,
                                                                          3)),
                layers.experimental.preprocessing.RandomRotation(0.1),
                layers.experimental.preprocessing.RandomZoom(0.1),
            ]
        )

        # Create model
        self.model = Sequential([
            data_augmentation,
            layers.experimental.preprocessing.Rescaling(1. / 255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes)
        ])

        # Compile model
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.summary()

    def load_latest_checkpoint(self):
        """Load the latest training weights

        """
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        print("\nLoading checkpoint: " + str(latest))
        if latest is not None:
            self.model.load_weights(latest)
        else:
            print("No checkpoint saved in: " + str(self.checkpoint_dir))

    def train_model(self):
        """Train the model and save a checkpoint after each epoch

        """
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         )

        # Train model
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.num_epochs,
            callbacks=[cp_callback]  # Pass callback to training
        )

        # self.plot_history()
        self.evaluate_model()

    def evaluate_model(self):
        """Print out the model's accuracy

        """
        loss, acc = self.model.evaluate(self.val_ds, verbose=2)
        print("\nModel accuracy: {:5.2f}%".format(100 * acc))

    def plot_history(self):
        """Plot training history

        """
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.num_epochs)

        plt.figure(figsize=(8, 8))
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

    def predict(self, img_path):
        """Make a prediction of the classification of an image

        """
        img = keras.preprocessing.image.load_img(
            img_path, target_size=(self.img_height, self.img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print("\nPrediction for: " + str(img_path))
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(self.class_names[np.argmax(score)], 100 * np.max(score))
        )
        return "This image most likely belongs to " + self.class_names[np.argmax(score)] + " with a " + str(
            self.truncate(100 * np.max(score))) + " percent confidence."

    def predict_all(self, path):
        """Make predictions on all images in a directory

        """
        files = [f for f in listdir(path) if isfile(join(path, f))]
        for f in files:
            print(self.predict(path + f))

    def truncate(self, n):
        """To truncate a number to two decimals

        """
        return int(n * 100) / 100


def flower_model():
    """The flower classifier

    """
    m = Model("./training_checkpoints/flowers/training_0/cp.ckpt")
    m.num_classes = 5
    m.class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    m.create_and_compile_model()
    m.load_latest_checkpoint()

    return m


def landscape_model():
    """The landscape classifier

    """
    m = Model("./training_checkpoints/landscapes/training_0/cp.ckpt")
    m.num_classes = 6
    m.class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    m.create_and_compile_model()
    m.load_latest_checkpoint()

    return m


if __name__ == '__main__':
    flower_model()
    landscape_model()
