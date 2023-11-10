## Machine Learning Models' Privacy Risk Evaluator


### For Tensorflow Models

`Must have Python version >= 3.9`  and  `install tensorflow_privacy library`
    
    Arguments for running  `main.py`
    
    --model_path  | MODEL_PATH  Absolute path where pretrained model is saved                            
    --attack      | ATTACK      Attack type: "custom" | "advanced"

For  `advanced attack` change the config file in attacks/config.py

    aconf = {
        'lr': 0.02,
        'batch_size': 128,
        'epochs': 2,
        'n_shadows': 2,
        'shpath': './attacks/shadows'
    }


### For Pytorch Models

    ## TODO
