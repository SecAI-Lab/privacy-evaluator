from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from scipy import special
from utils.data import AData

_ABSOLUTE_TOLERANCE = 1e-3


def _log_value(probs, small_value=1e-30):
    return -np.log(np.maximum(probs, small_value))


def get_entropy(logits, true_labels):
    """
    https://arxiv.org/pdf/2003.10595.pdf
    """

    if (np.absolute(np.sum(logits, axis=1) - 1) <= _ABSOLUTE_TOLERANCE).all():
        probs = logits
    else:
        probs = special.softmax(logits, axis=1)
    if true_labels is None:
        return np.sum(np.multiply(probs, _log_value(probs)), axis=1)
    else:

        log_probs = _log_value(probs)
        reverse_probs = 1 - probs
        log_reverse_probs = _log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size),
                       true_labels] = reverse_probs[range(true_labels.size),
                                                    true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size),
                           true_labels] = log_probs[range(true_labels.size),
                                                    true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)


def get_losses(model, tdata):
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

    return AData(
        train_loss=loss_train,
        test_loss=loss_test,
        train_logit=logits_train,
        test_logit=logits_train,
        train_labels=tdata.train_labels,
        test_labels=tdata.test_labels
    )


def get_ppv(tpr, fpr, test_train_ratio):
    """
     PPV=TP/(TP+FP)
     Revisiting Membership Inference Under Realistic Assumptions: https://arxiv.org/pdf/2005.10881.pdf
     src: 
    """

    num = np.asarray(tpr)
    den = num + np.asarray([r * test_train_ratio for r in fpr])
    tpr_is_0 = np.isclose(tpr, 0.0, atol=_ABSOLUTE_TOLERANCE)
    fpr_is_0 = np.isclose(fpr, 0.0, atol=_ABSOLUTE_TOLERANCE)
    tpr_and_fpr_both_0 = np.logical_and(tpr_is_0, fpr_is_0)

    ppv_when_tpr_fpr_both_0 = 1. / (1. + test_train_ratio)

    ppv_when_one_of_tpr_fpr_not_0 = np.divide(
        num, den, out=np.zeros_like(den), where=den != 0)
    return np.max(
        np.where(tpr_and_fpr_both_0, ppv_when_tpr_fpr_both_0,
                 ppv_when_one_of_tpr_fpr_not_0))
