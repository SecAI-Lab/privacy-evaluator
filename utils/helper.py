from tensorflow.keras.utils import to_categorical
import tensorflow as tf


def get_losses(model_path, tdata):
    model = tf.keras.models.load_model(model_path, compile=False)
    print("\nTesting on Test data....")
    logits_test = model.predict(tdata.test_data)

    print("\nTesting on Train data....")
    logits_train = model.predict(tdata.train_data)

    print('Apply softmax to get probabilities from logits...')
    prob_train = tf.nn.softmax(logits_train, axis=-1)
    prob_test = tf.nn.softmax(logits_test)

    print('Compute losses...')
    cce = tf.keras.backend.categorical_crossentropy
    constant = tf.keras.backend.constant

    y_train_onehot = to_categorical(tdata.train_labels)
    y_test_onehot = to_categorical(tdata.test_labels)

    loss_train = cce(constant(y_train_onehot), constant(
        prob_train), from_logits=False).numpy()
    loss_test = cce(constant(y_test_onehot), constant(
        prob_test), from_logits=False).numpy()

    return loss_train, loss_test
