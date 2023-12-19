from nobuco import ChannelOrder, pytorch_to_keras


def torch2tf(model, test_data):
    keras_model = nobuco.pytorch_to_keras(
        model,
        args=[test_data],
        inputs_channel_order=ChannelOrder.PYTORCH,
        outputs_channel_order=ChannelOrder.PYTORCH,
    )
    print(keras_model)
