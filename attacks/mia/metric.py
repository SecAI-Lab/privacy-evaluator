import numpy as np
import tensorflow as tf
from privacy_meter.audit import Audit, MetricEnum
from privacy_meter.audit_report import ROCCurveReport, SignalHistogramReport
from privacy_meter.constants import InferenceGame
from privacy_meter.dataset import Dataset
from privacy_meter.information_source import InformationSource

from _utils.wrapper_model import WrapperTF
from _utils.helper import get_trg_ref_data
from attacks.config import priv_meter as pm


def get_attack_data(tdata):
    x_train_all = np.concatenate(
        [data for data, _ in tdata.train_data], axis=0)
    x_test_all = np.concatenate([data for data, _ in tdata.test_data], axis=0)
    y_train_all = tdata.train_labels
    y_test_all = tdata.test_labels

    x_train, y_train = x_train_all[:pm['num_train_points']
                                   ], y_train_all[:pm['num_train_points']]
    x_test, y_test = x_test_all[:pm['num_test_points']
                                ], y_test_all[:pm['num_test_points']]
    x_population = x_train_all[pm['num_train_points']:(
        pm['num_train_points'] + pm['num_population_points'])]
    y_population = y_train_all[pm['num_train_points']:(
        pm['num_train_points'] + pm['num_population_points'])]

    tdata.train_data = x_train
    tdata.test_data = x_test
    tdata.train_labels = y_train
    tdata.test_labels = y_test
    tdata.x_concat = x_population
    tdata.y_concat = y_population

    target_dataset, reference_dataset = get_trg_ref_data(tdata)
    return tdata, target_dataset, reference_dataset


def run_population_metric(tdata, model, is_torch=False):
    tdata, target_dataset, reference_dataset = get_attack_data(tdata)
    if not is_torch:
        target_model = WrapperTF(model_obj=model, tdata=tdata)

    target_info_source = InformationSource(
        models=[target_model],
        datasets=[target_dataset]
    )
    reference_info_source = InformationSource(
        models=[target_model],
        datasets=[reference_dataset]
    )
    audit_obj = Audit(
        metrics=MetricEnum.POPULATION,
        inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
        target_info_sources=target_info_source,
        reference_info_sources=reference_info_source,
        fpr_tolerances=pm['fpr_tolerance_list']
    )
    print("Preparing Population Metric Attack....")
    audit_obj.prepare()
    print("Started Population Metric Attack....")
    audit_results = audit_obj.run()[0]
    print(audit_results[0])
