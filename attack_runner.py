import torch
import os


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
    from attacks.mia.lira import get_shadow_stats, run_advanced_attack
    from attacks.mia.pupulation import run_population_metric
    from attacks.mia.reference import run_reference_metric
    from target.tf_target import load_tf_cifar10
    from target.torch_target import load_torch_cifar10

    if is_torch:
        tdata = load_torch_cifar10()
    else:
        tdata = load_tf_cifar10()

    if attack == 'custom':
        attack_input = get_attack_inp(model, tdata, is_torch)
        run_custom_attacks(attack_input)

    elif attack == 'lira':
        adata = get_shadow_stats(model, tdata, is_torch)
        run_advanced_attack(adata)

    elif attack == 'population':
        run_population_metric(tdata, model, is_torch)

    elif attack == 'reference':
        run_reference_metric(tdata, model, is_torch)

    else:
        raise NotImplementedError('The other type of attacks not implemented!')
