import warnings
import tensorflow as tf
from _utils.helper import get_attack_inp
from attacks.mia.custom import run_custom_attacks
from attacks.mia.advanced import train_shadows, run_advanced_attack
from target.tf_target import load_cifar10
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    mpath = 'target/weights/cifar10_densenet.h5'
    model = tf.keras.models.load_model(mpath, compile=False)

    tdata = load_cifar10()
    train_shadows(model, tdata)
    # attack_input = get_attack_inp(model, tdata)

    # run_custom_attacks(attack_input)
