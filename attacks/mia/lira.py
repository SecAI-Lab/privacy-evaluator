from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import plotting as mia_plotting
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import advanced_mia as amia
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import functools
import torch
import os
import gc

from _utils.helper import get_stat_and_loss_aug, plot_curve_with_area
from _utils.data import AdvAttackData
from attacks.config import aconf
from target.tf_target import train
from target.torch_target import torch_train

seed = 123
np.random.seed(seed)


def get_shadow_stats(model, tdata, is_torch=False):
    x = tdata.x_concat
    y = tdata.y_concat
    n = x.shape[0]
    sample_weight = None
    ext = 'h5'
    in_indices, stat, losses = [], [], []
    os.makedirs(aconf['shpath'], exist_ok=True)

    if is_torch:
        ext = 'pt'

    for i in range(aconf['n_shadows']):
        in_indices.append(np.random.binomial(1, 0.5, n).astype(bool))
        model_path = os.path.join(
            aconf['shpath'],  f'model{i}_e{aconf["epochs"]}_sd{seed}.{ext}'
        )

        if os.path.exists(model_path):
            if is_torch:
                model.load_state_dict(torch.load(model_path))
            else:
                model(x[:1])
                model.load_weights(model_path)
            print(f'Loaded model #{i} with {in_indices[-1].sum()} examples.')

        else:
            tdata.train_data = x[in_indices[-1]]
            tdata.train_labels = y[in_indices[-1]]
            tdata.test_data = x[~in_indices[-1]]
            tdata.test_labels = y[~in_indices[-1]]

            if is_torch:
                torch_train(model, tdata, model_path)
            else:
                train(model_path, tdata=tdata, model=model)
            print(f'Trained model #{i} with {in_indices[-1].sum()} examples.')

        s, l = get_stat_and_loss_aug(
            model, x, y, sample_weight, is_torch=is_torch)

        stat.append(s)
        losses.append(l)
        tf.keras.backend.clear_session()
        gc.collect()

    return AdvAttackData(
        stat=stat,
        in_indices=in_indices,
        sample_weight=sample_weight,
        losses=losses,
        n=n
    )


def run_advanced_attack(attack_data):
    for idx in range(aconf['n_shadows']):
        print(f'\nTarget model is #{idx}')
        stat_target = attack_data.stat[idx]
        in_indices_target = attack_data.in_indices[idx]
        stat_shadow = np.array(
            attack_data.stat[:idx] + attack_data.stat[idx + 1:])
        in_indices_shadow = np.array(
            attack_data.in_indices[:idx] + attack_data.in_indices[idx + 1:])
        stat_in = [stat_shadow[:, j][in_indices_shadow[:, j]]
                   for j in range(attack_data.n)]
        stat_out = [stat_shadow[:, j][~in_indices_shadow[:, j]]
                    for j in range(attack_data.n)]

        scores = amia.compute_score_lira(
            stat_target, stat_in, stat_out, fix_variance=True)

        attack_input = AttackInputData(
            loss_train=scores[in_indices_target],
            loss_test=scores[~in_indices_target],
            sample_weight_train=attack_data.sample_weight,
            sample_weight_test=attack_data.sample_weight)
        result_lira = mia.run_attacks(attack_input).single_attack_results[0]

        print('\nAdvanced MIA attack with Gaussian:',
              f'auc = {result_lira.get_auc():.4f}',
              f'adv = {result_lira.get_attacker_advantage():.4f}')

        scores = -amia.compute_score_offset(stat_target, stat_in, stat_out)
        attack_input = AttackInputData(
            loss_train=scores[in_indices_target],
            loss_test=scores[~in_indices_target],
            sample_weight_train=attack_data.sample_weight,
            sample_weight_test=attack_data.sample_weight)
        result_offset = mia.run_attacks(attack_input).single_attack_results[0]

        print('\nAdvanced MIA attack with offset:',
              f'auc = {result_offset.get_auc():.4f}',
              f'adv = {result_offset.get_attacker_advantage():.4f}')

        loss_target = attack_data.losses[idx][:, 0]
        attack_input = AttackInputData(
            loss_train=loss_target[in_indices_target],
            loss_test=loss_target[~in_indices_target],
            sample_weight_train=attack_data.sample_weight,
            sample_weight_test=attack_data.sample_weight)
        result_baseline = mia.run_attacks(
            attack_input).single_attack_results[0]

        print('\nBaseline MIA attack:', f'auc = {result_baseline.get_auc():.4f}',
              f'adv = {result_baseline.get_attacker_advantage():.4f}')

    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    for res, title in zip([result_baseline, result_lira, result_offset],
                          ['baseline', 'LiRA', 'offset']):
        label = f'{title} auc={res.get_auc():.4f}'
        mia_plotting.plot_roc_curve(
            res.roc_curve,
            functools.partial(plot_curve_with_area, ax=ax, label=label))
    plt.legend()
    plt.savefig('advanced_mia.png')
