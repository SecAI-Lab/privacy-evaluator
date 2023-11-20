## Privacy Risk Assessment tool for Machine Learning Models


### For Tensorflow (.h5) and Pytorch (.pt) Models

`Must have Python version >= 3.9`  and  `install tensorflow_privacy library`
    
    Arguments for running  `main.py`
    
    --model_path  | MODEL_PATH  Absolute path where pretrained model is saved                            
    --attack      | ATTACK      Attack type: "custom" | "lira" | "population" | "reference"

> Note, the pretrained model should be saved as a whole, not only `state_dict` format which requires model initialization!

For `lira attacks` you can change the config file in `attacks/config.py`

    aconf = {
        'lr': 0.02,
        'batch_size': 128,
        'epochs': 2,
        'n_shadows': 2,
        'shpath': './attacks/shadows'
    }

For `population` and `reference` metric attacks same config file as above but:

    priv_meter = {
        'num_classes': 10,
        'num_train_points': 5000,
        'num_test_points': 5000,
        'epochs': 10,
        'batch_size': 64,
        'num_population_points': 10000,
        'fpr_tolerance_list': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'input_shape': (224, 224, 3),
        'ref_models': './attacks/shadows/',
        'torch_loss': torch.nn.CrossEntropyLoss(),
        'tf_loss': tf.keras.losses.CategoricalCrossentropy()
    }

parameters can be updated.