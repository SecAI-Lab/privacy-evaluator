from privacy_meter.model import TensorflowModel, PytorchModel
import tensorflow as tf
import torch

from target.torch_target import torch_predict


class WrapperTF(TensorflowModel):
    def __init__(self, model_obj, loss_fn):
        super().__init__(model_obj, loss_fn)
        self.model = model_obj
        self.loss_fn = loss_fn

    def get_logits(self, batch_samples):
        logits = self.model.predict(batch_samples)
        prob = tf.nn.softmax(logits, axis=-1)
        return prob

    def get_loss(self, batch_samples, batch_labels, per_point=True):
        logits = self.get_logits(batch_samples)
        # prob = tf.nn.softmax(logits, axis=-1)
        losses = self.loss_fn(logits, batch_labels)
        return losses


class WrapperTorch(PytorchModel):
    def __init__(self, model_obj, loss_fn):
        super().__init__(model_obj, loss_fn)
        self.model = model_obj
        self.loss_fn = loss_fn

    def get_logits(self, batch_samples):
        return torch_predict(self.model, batch_samples)

    def get_loss(self, batch_samples, batch_labels, per_point=True):
        logits = self.get_logits(batch_samples)
        logits = torch.from_numpy(logits)
        batch_labels = torch.from_numpy(batch_labels)
        losses = self.loss_fn(logits, batch_labels)
        return losses
