from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
from dataclasses import dataclass
from typing import Any

"""Test file just to prepare TF model"""


@dataclass
class TData:
    train_data: Any
    train_labels: Any
    test_data: Any
    test_labels: Any


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (224, 224))
    return image, label


def load_cifar10():
    (train_data, train_labels), (test_data,
                                 test_labels) = tf.keras.datasets.cifar10.load_data()
    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels))

    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    train_data = (train_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=100, drop_remainder=False))
    test_data = (test_ds
                 .map(process_images)
                 .shuffle(buffer_size=test_ds_size)
                 .batch(batch_size=10, drop_remainder=False))

    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    return TData(
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels
    )


def densenet(num_classes):
    base_model = DenseNet121(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes)(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    return model


def train():
    tdata = load_cifar10()
    checkpoint_path = 'weights/cifar10_densenet.h5'
    model = densenet(num_classes=10)

    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(tdata.train_data,
              validation_data=tdata.test_data,
              batch_size=64,
              epochs=20)
    model.save(checkpoint_path)
