from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import advanced_mia as amia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import plotting as mia_plotting
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
import matplotlib.pyplot as plt
from attacks.config import aconf
from _utils.helper import get_stat_and_loss_aug
import numpy as np
import tensorflow as tf
import os


def train_shadows(model, tdata):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x = np.concatenate([x_train, x_test]).astype(np.float32) / 255
    y = np.concatenate([y_train, y_test]).astype(np.int32).squeeze()
    
    seed = 123
    np.random.seed(seed)
    
    sample_weight = None
    n = x.shape[0]

    in_indices = []  
    stat = []  
    losses = [] 
    
    for i in range(aconf['n_shadows'] + 1):
        
        if aconf['shpath']:
            model_path = os.path.join(
                aconf['shpath'],
                f'model{i}_lr{aconf["lr"]}_b{aconf["batch_size"]}_e{aconf["epochs"]}_sd{seed}.h5'
            )

        in_indices.append(np.random.binomial(1, 0.5, n).astype(bool))
        
        if aconf['shpath'] and os.path.exists(model_path):  
            model(x[:1]) 
            model.load_weights(model_path)
            print(f'Loaded model #{i} with {in_indices[-1].sum()} examples.')
        
        else:  
            model.compile(
                optimizer=tf.keras.optimizers.SGD(aconf['lr'], momentum=0.9),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
            model.fit(
                x[in_indices[-1]],
                y[in_indices[-1]],
                validation_data=(x[~in_indices[-1]], y[~in_indices[-1]]),
                epochs=aconf['epochs'],
                batch_size=aconf['batch_size'],
                verbose=2)
            if aconf['shpath']:
                model.save_weights(model_path)
            print(f'Trained model #{i} with {in_indices[-1].sum()} examples.')
       
        s, l = get_stat_and_loss_aug(model, x, y, sample_weight)
        stat.append(s)
        losses.append(l)
        tf.keras.backend.clear_session()
        gc.collect()


def run_advanced_attack():
    for idx in range(aconf['n_shadows'] + 1):
        print(f'Target model is #{idx}')
        stat_target = stat[idx]  
        in_indices_target = in_indices[idx] 
        stat_shadow = np.array(stat[:idx] + stat[idx + 1:])
        in_indices_shadow = np.array(in_indices[:idx] + in_indices[idx + 1:])
        stat_in = [stat_shadow[:, j][in_indices_shadow[:, j]] for j in range(n)]
        stat_out = [stat_shadow[:, j][~in_indices_shadow[:, j]] for j in range(n)]
       
        scores = amia.compute_score_lira(
            stat_target, stat_in, stat_out, fix_variance=True)
        
        attack_input = AttackInputData(
            loss_train=scores[in_indices_target],
            loss_test=scores[~in_indices_target],
            sample_weight_train=sample_weight,
            sample_weight_test=sample_weight)
        result_lira = mia.run_attacks(attack_input).single_attack_results[0]
        
        print('Advanced MIA attack with Gaussian:',
            f'auc = {result_lira.get_auc():.4f}',
            f'adv = {result_lira.get_attacker_advantage():.4f}')

        scores = -amia.compute_score_offset(stat_target, stat_in, stat_out)
        attack_input = AttackInputData(
            loss_train=scores[in_indices_target],
            loss_test=scores[~in_indices_target],
            sample_weight_train=sample_weight,
            sample_weight_test=sample_weight)
        result_offset = mia.run_attacks(attack_input).single_attack_results[0]
        
        print('Advanced MIA attack with offset:',
            f'auc = {result_offset.get_auc():.4f}',
            f'adv = {result_offset.get_attacker_advantage():.4f}')

        loss_target = losses[idx][:, 0]
        attack_input = AttackInputData(
            loss_train=loss_target[in_indices_target],
            loss_test=loss_target[~in_indices_target],
            sample_weight_train=sample_weight,
            sample_weight_test=sample_weight)
        result_baseline = mia.run_attacks(attack_input).single_attack_results[0]
        
        print('Baseline MIA attack:', f'auc = {result_baseline.get_auc():.4f}',
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