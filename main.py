import warnings
import tensorflow as tf
from _utils.helper import get_attack_inp, is_valid
from attacks.mia.custom import run_custom_attacks
from attacks.mia.advanced import train_shadows, run_advanced_attack
from target.tf_target import load_cifar10, train
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    arg = sys.argv[-1]  # option 1: custom attack;  option 2: advanced attack

    mpath = 'target/weights/cifar10_densenet.h5'

    if not is_valid(mpath):
        train(mpath)

    model = tf.keras.models.load_model(mpath, compile=False)
    tdata = load_cifar10()

    if arg == 1:
        attack_input = get_attack_inp(model, tdata)
        run_custom_attacks(attack_input)
    elif arg == 2:
        adata = train_shadows(model, tdata)
        run_advanced_attack(adata)
