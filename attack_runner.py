import numpy as np
import torch
import os

np.random.seed(1234)


def runner(args):
    mpath = args.model_path
    attack = args.attack
    n_class = int(args.n_class)
    is_torch = False

    if args.train:
        if mpath.endswith('.h5'):
            from target.tf_target import train
            train(checkpoint_path=mpath, with_dp=args.dp_on)
        else:
            from target.torch_target import torch_train
            torch_train(checkpoint_path=mpath)

    if not os.path.exists(mpath):
        raise FileExistsError("Model path doesn't exist!")

    if mpath.endswith('.h5'):
        import tensorflow as tf
        model = tf.keras.models.load_model(mpath, compile=False)
    elif mpath.endswith('.pt') or mpath.endswith('.pth'):
        model = torch.load(mpath)
        if isinstance(model, dict):
            print(model.keys())
            if isinstance(model.get('state'), dict):
                print("Model state dict passed! Need a whole model object.")
                exit(0)
            else:
                model = model['state']
        is_torch = True

    from _utils.helper import get_attack_inp
    from attacks.mia.custom import run_custom_attacks
    from attacks.mia.lira import run_advanced_attack
    from attacks.mia.population import run_population_metric
    from attacks.mia.reference import run_reference_metric
    from attacks.mia.shadow import run_shadow_metric
    from target.tf_target import load_tf_cifar
    from target.torch_target import load_torch_cifar

    if is_torch:
        tdata = load_torch_cifar(num_class=n_class)
    else:
        tdata = load_tf_cifar(num_class=n_class)

    if attack == 'custom':
        attack_input = get_attack_inp(model, tdata, is_torch)
        run_custom_attacks(attack_input)

    elif attack == 'lira':
        run_advanced_attack(model, n_class, tdata, is_torch)

    elif attack == 'population':
        run_population_metric(tdata, model, n_class, is_torch)

    elif attack == 'reference':
        run_reference_metric(tdata, model, n_class, is_torch)

    elif attack == 'shadow':
        run_shadow_metric(tdata, model, n_class, is_torch)

    else:
        raise NotImplementedError('The other type of attacks not implemented!')
