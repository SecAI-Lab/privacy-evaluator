import tensorflow as tf
import os

from _utils.helper import get_attack_inp, is_valid
from attacks.mia.custom import run_custom_attacks
from attacks.mia.advanced import train_shadows, run_advanced_attack
from target.tf_target import load_cifar10, train


def runner(args):
    mpath = args.model_path
    attack = args.attack

    if not os.path.exists(mpath):
        raise FileExistsError("Model path doesn't exist!")

    model = tf.keras.models.load_model(mpath, compile=False)
    tdata = load_cifar10()

    if attack == 'custom':
        attack_input = get_attack_inp(model, tdata)
        run_custom_attacks(attack_input)

    elif attack == 'advanced':
        adata = train_shadows(model, tdata)
        run_advanced_attack(adata)

    else:
        raise NotImplementedError('The other type of attacks not implemented!')
