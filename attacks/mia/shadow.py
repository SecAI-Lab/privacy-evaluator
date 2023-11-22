from privacy_meter.information_source import InformationSource
from privacy_meter.constants import InferenceGame
from privacy_meter.audit import Audit, MetricEnum
import torch
import os

from _utils.wrapper_model import WrapperTF, WrapperTorch
from _utils.helper import get_trg_ref_data
from attacks.config import priv_meter as pm


def run_shadow_metric(tdata, model, is_torch):
    target_dataset = get_trg_ref_data(tdata)
    shadow_path = os.listdir(pm['ref_models'])
    shadow_models = []

    if is_torch:
        target_model = WrapperTorch(model_obj=model, loss_fn=pm['torch_loss'])
    else:
        target_model = WrapperTF(model_obj=model, loss_fn=pm['tf_loss'])

    for shadow in shadow_path:
        fpath = pm['ref_models'] + shadow
        if is_torch and fpath.endswith('.pt'):
            model.load_state_dict(torch.load(fpath))
            shadow_models.append(WrapperTorch(
                model_obj=model, loss_fn=pm['torch_loss']))

        elif not is_torch and fpath.endswith('.h5'):
            model.load_weights(fpath)
            shadow_models.append(
                WrapperTF(model_obj=model, loss_fn=pm['tf_loss']))

    datasets_list = target_dataset.subdivide(
        num_splits=pm['n_shadows'],
        return_results=True,
        split_size=pm['num_train_points']
    )

    target_info_source = InformationSource(
        models=[target_model],
        datasets=[target_dataset]
    )

    reference_info_source = InformationSource(
        models=shadow_models,
        datasets=datasets_list
    )

    audit = Audit(
        metrics=MetricEnum.SHADOW,
        inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
        target_info_sources=target_info_source,
        reference_info_sources=reference_info_source,
        # save_logs=False
    )
    print("Preparing shadow metric attack....")
    audit.prepare()

    print("Starting shadow metric attack....")
    result = audit.run()[0]
    print(result)