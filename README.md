## Privacy Risk Assessment tool for Machine Learning Models


### For Tensorflow (.h5) and Pytorch (.pt) Models

`Must have Python version >= 3.9`  and  `install tensorflow_privacy library`
    
    Arguments for running  `main.py`
    
    --model_path  | MODEL_PATH  Absolute path where pretrained model is saved                            
    --attack      | ATTACK      Attack type: "custom" | "lira"

For `lira attack` you can change the config file in `attacks/config.py`

    aconf = {
        'lr': 0.02,
        'batch_size': 128,
        'epochs': 2,
        'n_shadows': 2,
        'shpath': './attacks/shadows'
    }


