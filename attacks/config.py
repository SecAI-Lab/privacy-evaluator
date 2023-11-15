aconf = {
    'lr': 0.01,
    'batch_size': 64,
    'epochs': 20,
    'n_shadows': 2,
    'shpath': './attacks/shadows'
}

priv_meter = {
    'num_train_points': 5000,
    'num_test_points': 5000,
    'epochs': 10,
    'batch_size': 64,
    'num_population_points': 10000,
    'fpr_tolerance_list': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'input_shape': (224, 224, 3)
}
