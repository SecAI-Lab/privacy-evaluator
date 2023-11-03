import warnings
import tensorflow as tf
# from attacks.mia.threshold import run_threshold_attack
# from attacks.mia.single_entropy import run_threshold_entropy_attack
from target.tf_target import load_cifar10
# from utils.helper import get_losses
from tensorflow.keras.utils import to_categorical
import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    mpath = 'target/weights/cifar10_densenet.h5'
    model = tf.keras.models.load_model(mpath, compile=False)
    tdata = load_cifar10()
    # adata = get_losses(model, tdata)

    # # run_threshold_attack(adata.loss_train, adata.loss_test)
    # run_threshold_entropy_attack(adata)

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

    slicing_spec = SlicingSpec(
        entire_dataset=True,
        by_class=True,
        by_percentiles=False,
        by_classification_correctness=True
    )

    attack_types = [
        AttackType.THRESHOLD_ATTACK,
        AttackType.LOGISTIC_REGRESSION
    ]

    attacks_result = mia.run_attacks(attack_input=attack_input,
                                     slicing_spec=slicing_spec,
                                     attack_types=attack_types)

    print(attacks_result.summary(by_slices=False))
    print('\n----------------------------\n')
    print(attacks_result.summary(by_slices=True))
