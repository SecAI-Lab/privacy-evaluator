import os
import torch


def runner(args):
    mpath = args.model_path
    attack = args.attack
    is_torch = False

    if not os.path.exists(mpath):
        raise FileExistsError("Model path doesn't exist!")

    if mpath.endswith('.h5'):
        import tensorflow as tf
        model = tf.keras.models.load_model(mpath, compile=False)
    elif mpath.endswith('.pt'):
        model = torch.load(mpath)
        is_torch = True

    from _utils.helper import get_attack_inp
    from attacks.mia.custom import run_custom_attacks
    from attacks.mia.advanced import train_shadows, run_advanced_attack
    from target.tf_target import load_tf_cifar10
    from target.torch_target import load_torch_cifar10

    if is_torch:
        tdata = load_torch_cifar10()
    else:
        tdata = load_tf_cifar10()

    if attack == 'custom':
        attack_input = get_attack_inp(model, tdata, is_torch)
        run_custom_attacks(attack_input)

    elif attack == 'advanced':
        adata = train_shadows(model, tdata)
        run_advanced_attack(adata, is_torch)

    else:
        raise NotImplementedError('The other type of attacks not implemented!')
