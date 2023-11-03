from sklearn import metrics
import numpy as np
import json
from utils.helper import get_ppv


def run_threshold_attack(loss_train, loss_test):
    ntrain, ntest = loss_train.shape[0], loss_test.shape[0]

    fpr, tpr, _ = metrics.roc_curve(
        np.concatenate((np.ones(ntrain), np.zeros(ntest))),
        -np.concatenate((loss_train, loss_test)),
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
    print('\nThreshold Attack Result:\n', json.dumps(result_dict, indent=2))
