import argparse
from attack_runner import runner

if __name__ == '__main__':
    mpath = 'target/weights/cifar10_densenet.h5'
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', default=mpath,
                        help='an absolute path where pretrained model is saved')
    parser.add_argument('--attack', default='custom',
                        help='An attack type: "custom" | "advanced" ')
    args = parser.parse_args()
    print(args)
    runner(args)
