from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import advanced_mia as amia
from tensorflow_privacy.privacy.privacy_tests import utils
from tensorflow.keras.utils import to_categorical
from typing import Optional
import tensorflow as tf
import numpy as np


def get_attack_inp(model, tdata):    
    print('Predict on train...')
    logits_train = model.predict(tdata.train_data)
    print('Predict on test...')
    logits_test = model.predict(tdata.test_data)

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

    attack_input = AttackInputData(
        logits_train=logits_train,
        logits_test=logits_test,
        loss_train=loss_train,
        loss_test=loss_test,
        labels_train=tdata.train_labels,
        labels_test=tdata.test_labels
    )
    return attack_input


def get_stat_and_loss_aug(model,
                          x,
                          y,
                          sample_weight: Optional[np.ndarray] = None,
                          batch_size=64):
  
  losses, stat = [], []
  for data in [x, x[:, :, ::-1, :]]:
    prob = amia.convert_logit_to_prob(
        model.predict(data, batch_size=batch_size))
    losses.append(utils.log_loss(y, prob, sample_weight=sample_weight))
    stat.append(
        amia.calculate_statistic(
            prob, y, sample_weight=sample_weight, is_logits=False))
  return np.vstack(stat).transpose(1, 0), np.vstack(losses).transpose(1, 0)