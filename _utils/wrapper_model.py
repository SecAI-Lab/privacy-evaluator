from privacy_meter.model import TensorflowModel
import tensorflow as tf

from _utils.helper import get_attack_inp


class WrapperTF(TensorflowModel):
    def __init__(self, model_obj, tdata=None):
        super().__init__(model_obj, loss_fn=tf.keras.losses.CategoricalCrossentropy())
        self.tdata = tdata
        self.adata = None
        self.model = model_obj

    def get_logits(self, batch_samples):
        return self.adata.logits_train

    def get_loss(self, batch_samples, batch_labels, per_point=True):
        self.tdata.train_data = batch_samples
        self.tdata.train_labels = batch_labels
        self.adata = get_attack_inp(self.model, self.tdata, is_torch=False)
        return self.adata.loss_train
