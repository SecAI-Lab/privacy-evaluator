import numpy as np
from sklearn import metrics
import json

_ABSOLUTE_TOLERANCE = 1e-3


def get_ppv(tpr, fpr, test_train_ratio):
    """
     PPV=TP/(TP+FP)
     Revisiting Membership Inference Under Realistic Assumptions: https://arxiv.org/pdf/2005.10881.pdf
    """

    num = np.asarray(tpr)
    den = num + np.asarray([r * test_train_ratio for r in fpr])
    tpr_is_0 = np.isclose(tpr, 0.0, atol=_ABSOLUTE_TOLERANCE)
    fpr_is_0 = np.isclose(fpr, 0.0, atol=_ABSOLUTE_TOLERANCE)
    tpr_and_fpr_both_0 = np.logical_and(tpr_is_0, fpr_is_0)

    ppv_when_tpr_fpr_both_0 = 1. / (1. + test_train_ratio)

    ppv_when_one_of_tpr_fpr_not_0 = np.divide(
        num, den, out=np.zeros_like(den), where=den != 0)
    return np.max(
        np.where(tpr_and_fpr_both_0, ppv_when_tpr_fpr_both_0,
                 ppv_when_one_of_tpr_fpr_not_0))


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
