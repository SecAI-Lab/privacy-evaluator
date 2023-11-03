from sklearn import metrics
import numpy as np
from utils.helper import *
import json


def run_threshold_entropy_attack(attack_input):
    ntrain, ntest = attack_input.train_logit.shape[0], attack_input.test_logit.shape[0]
    train_entropy = get_entropy(
        attack_input.train_logit, attack_input.train_labels)
    test_entropy = get_entropy(
        attack_input.test_logit, attack_input.test_labels)
    fpr, tpr, _ = metrics.roc_curve(
        np.concatenate((np.ones(ntrain), np.zeros(ntest))),
        -np.concatenate(
            (train_entropy, test_entropy)
        ),
    )

    test_train_ratio = ntest / ntrain

    auc = metrics.auc(fpr, tpr)
    adv = max(np.abs(tpr - fpr))
    ppv = get_ppv(tpr, fpr, test_train_ratio)
    result_dict = {
        'AUC': auc,
        'Attacker advantage': adv,
        'Positive Predictive Value of attacker': ppv
    }
    print('\nThreshold Entropy Attack Result:\n',
          json.dumps(result_dict, indent=2))
