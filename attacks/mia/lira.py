from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import advanced_mia as amia
import tensorflow as tf
import numpy as np
import torch
import os
import gc

from _utils.helper import get_stat_and_loss_aug
from _utils.data import ShadowStats
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
        # TODO: change model loading without initilization
        if os.path.exists(model_path):
            if is_torch:
                model.load_state_dict(torch.load(model_path))
            else:
                model(x[:1])
                model.load_weights(model_path)
            print(
                f'\nLoaded shadow model #{i} with {in_indices[-1].sum()} examples.')

        else:
            tdata.train_data = x[in_indices[-1]]
            tdata.train_labels = y[in_indices[-1]]
            tdata.test_data = x[~in_indices[-1]]
            tdata.test_labels = y[~in_indices[-1]]

            if is_torch:
                torch_train(model, tdata, model_path)
            else:
                train(model_path, tdata=tdata, pretrained=model)
            print(f'Trained model #{i} with {in_indices[-1].sum()} examples.')

        s, l = get_stat_and_loss_aug(
            model, x, y, is_torch=is_torch)
        stat.append(s)
        losses.append(l)

    tf.keras.backend.clear_session()
    gc.collect()

    return ShadowStats(
        stat=stat,
        in_indices=in_indices,
        sample_weight=sample_weight,
        losses=losses,
        n=n
    )


def run_advanced_attack(model, tdata, is_torch):
    shdata = get_shadow_stats(model, tdata, is_torch)

    stat_target_train, loss_target_train = get_stat_and_loss_aug(
        model, tdata.train_data, tdata.train_labels, is_torch=is_torch)
    stat_target_test, loss_target_test = get_stat_and_loss_aug(
        model, tdata.test_data, tdata.test_labels, is_torch=is_torch)

    train_len = stat_target_train.shape[0]
    test_len = stat_target_test.shape[0]

    for idx in range(aconf['n_shadows']):
        stat_shadow = np.array(shdata.stat[:idx] + shdata.stat[idx + 1:])
        in_indices_shadow = np.array(
            shdata.in_indices[:idx] + shdata.in_indices[idx + 1:])
        stat_in = [stat_shadow[:, j][in_indices_shadow[:, j]]
                   for j in range(shdata.n)]
        stat_out = [stat_shadow[:, j][~in_indices_shadow[:, j]]
                    for j in range(shdata.n)]

        scores_in = amia.compute_score_lira(
            stat_target_train, stat_in[:train_len], stat_out[:train_len], fix_variance=True)
        scores_out = amia.compute_score_lira(
            stat_target_test, stat_in[:test_len], stat_out[:test_len], fix_variance=True)

        attack_input = AttackInputData(
            loss_train=scores_in,
            loss_test=scores_out,
            sample_weight_train=shdata.sample_weight,
            sample_weight_test=shdata.sample_weight)
        result_lira = mia.run_attacks(attack_input).single_attack_results[0]

        print('\nLiRA attack with Gaussian:',
              f'auc = {result_lira.get_auc():.4f}',
              f'adv = {result_lira.get_attacker_advantage():.4f}')

        # Computing LiRA offset
        scores_in = - \
            amia.compute_score_offset(
                stat_target_train, stat_in[:train_len], stat_out[:train_len])
        scores_out = - \
            amia.compute_score_offset(
                stat_target_test, stat_in[:test_len], stat_out[:test_len])
        attack_input.loss_train = scores_in
        attack_input.loss_test = scores_out

        result_offset = mia.run_attacks(attack_input).single_attack_results[0]

        print('\nLiRA attack with offset:',
              f'auc = {result_offset.get_auc():.4f}',
              f'adv = {result_offset.get_attacker_advantage():.4f}')

        # Computing LiRA baseline
        attack_input.loss_train = loss_target_train.flatten()
        attack_input.loss_test = loss_target_test.flatten()

        result_baseline = mia.run_attacks(
            attack_input).single_attack_results[0]

        print('\nLiRA baseline attack:',
              f'auc = {result_baseline.get_auc():.4f}',
              f'adv = {result_baseline.get_attacker_advantage():.4f}')
