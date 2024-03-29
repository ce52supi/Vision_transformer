import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import cv2


class My_Custom_Generator(keras.utils.Sequence):

    def __init__(self, database_dir, image_filenames, labels, batch_size, input_shape=(50, 50, 1)):
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
            temp = cv2.imread(self.database_dir + str(file_name), cv2.IMREAD_GRAYSCALE)
            X = np.array(cv2.resize(temp, (50, 50)))
            X = np.expand_dims(X, axis=-1)
            x_train[idx, :, :, :] = X
            idx = idx + 1
        y = np.reshape(np.array(batch_y), (self.batch_size, 1))
        return x_train, y


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    print(inputs.shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            'accuracy',
            #keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    history = model.fit(my_training_batch_generator,
                   epochs = 30,
                   verbose = 1,
                   validation_data = my_validation_batch_generator
    )

    _, accuracy, top_5_accuracy = model.evaluate_generator(my_validation_batch_generator)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    model.save("30epochs_with_normalization")
    return history

image_data = pd.read_csv("class_balanced.csv")
X = image_data['image_name']
y = image_data['labels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

batch_size = 300
images_path = r"sushi/"
num_classes = 1
input_shape = (50, 50, 1)

learning_rate = 0.001
weight_decay = 0.0001
image_size = 50  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]


data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Normalization(),
        layers.experimental.preprocessing.Resizing(image_size, image_size),
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(factor=0.02),
    ],
    name="data_augmentation",
)

vit_classifier = create_vit_classifier()

vit_classifier.summary()
my_training_batch_generator = My_Custom_Generator(images_path, X_train, y_train, batch_size)
my_validation_batch_generator = My_Custom_Generator(images_path, X_test, y_test, batch_size)

run_experiment(vit_classifier)
