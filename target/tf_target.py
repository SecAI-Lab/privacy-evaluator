from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

from attacks.config import aconf
from _utils.data import TData

"""Test file for preparing TF model"""


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (224, 224))
    return image, label


def load_tf_cifar10():
    (train_data, train_labels), (test_data,
                                 test_labels) = tf.keras.datasets.cifar10.load_data()

    x = np.concatenate([train_data, test_data]).astype(np.float32) / 255
    y = np.concatenate([train_labels, test_labels]).astype(np.int32).squeeze()

    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()

    train_ds = (train_ds
                .map(process_images)
                .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=100, drop_remainder=False))
    test_ds = (test_ds
               .map(process_images)
               .shuffle(buffer_size=test_ds_size)
               .batch(batch_size=10, drop_remainder=False))

    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    return TData(
        train_data=train_ds,
        train_labels=train_labels,
        test_data=test_ds,
        test_labels=test_labels,
        x_concat=x,
        y_concat=y
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


def train(checkpoint_path, tdata=None, model=None):
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=aconf['lr'], momentum=0.9)

    if tdata is None:
        tdata = load_tf_cifar10()

    if model is None:
        model = densenet(num_classes=10)
        optimizer = tf.keras.optimizers.Adam()

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(tdata.train_data, tdata.train_labels,
              validation_data=(tdata.test_data, tdata.test_labels),
              batch_size=aconf['batch_size'],
              epochs=aconf['epochs'])

    if model == None:
        print("Saving whole model...")
        model.save(checkpoint_path)
    else:
        print("Saving model weights...")
        model.save_weights(checkpoint_path)
