from attacks.mia.threshold import run_threshold_attack
from target.tf_target import load_cifar10
from utils.helper import get_losses

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    mpath = 'target/weights/cifar10_resnet.h5'
    tdata = load_cifar10()
    loss_train, loss_test = get_losses(mpath, tdata)

    run_threshold_attack(loss_train, loss_test)
