import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import cv2


class My_Custom_Generator(keras.utils.Sequence):

    def __init__(self, database_dir, image_filenames, labels, batch_size, input_shape=(50, 50, 3)):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.database_dir = database_dir
        self.input_shape = input_shape

    def __len__(self):
        return (int(np.floor(len(self.image_filenames) / self.batch_size)))

    def __getitem__(self, idx):
        x_train = np.empty((self.batch_size, *self.input_shape))
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        list = []
        idx = 0
        for file_name in batch_x:
            temp = cv2.imread(self.database_dir + str(file_name))
            X = np.array(cv2.resize(temp, (50, 50)))
            X = np.expand_dims(X,axis=0)
            x_train[idx, :, :, :] = X
            idx = idx + 1
        y = np.reshape(np.array(batch_y), (self.batch_size, 1))
        return x_train, y


image_data = pd.read_csv("imbalance_removed.csv")
X = image_data['image_name']
y = image_data['labels']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

batch_size = 2000
images_path = r"sushi/"

my_training_batch_generator = My_Custom_Generator(images_path, X, y, batch_size)
model = keras.models.load_model("30epochs_Cross_validation")

score = model.predict(my_training_batch_generator)
predictions = keras.activations.sigmoid(score).numpy()

from sklearn.metrics import f1_score, accuracy_score

predictions[predictions < 0.5] = 0
predictions[predictions >= 0.5] = 1

print(f1_score(predictions, y))
print(accuracy_score(predictions, y))

