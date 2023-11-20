import warnings
import argparse
import pprint
import os

from attack_runner import runner

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    mpath = 'target/weights/cifar10.h5'
    parser = argparse.ArgumentParser(
        description='Arguments for attack configuration.')
    parser.add_argument('--model_path', default=mpath,
                        help='Absolute path where pretrained model is saved')
    parser.add_argument('--attack', default='custom',
                        help='Attack type: "custom" | "lira" | "population" | "reference" ')
    args = parser.parse_args()

    pprint.pprint(args)

    runner(args)
