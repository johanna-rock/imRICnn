import math
import torch
import numpy as np
from data_models.parameter_configuration import ParameterConfiguration
from datasets.radar_dataset import RadarDataset
from run_scripts import task_id
from utils.distribution import loguniform


def select_and_sample_hyperparameter_config_for_cnn(configurations):

    conf = configurations[task_id % len(configurations)]
    hyperparameter_config = ParameterConfiguration(
                            optimization_algo=torch.optim.Adam,
                            criterion=conf['criterion'],
                            scheduler_partial=None,
                            num_model_initializations=1,
                            scaler=conf['scaler'],
                            input_size=2028,
                            output_size=2048,
                            num_epochs=conf['num_epochs'],
                            input_data_source=conf['data_source'],
                            mat_path=conf['mat_path'],
                            model=conf['model'](num_conv_layer=conf['num_conv_layer'], num_filters=conf['num_filters'], filter_size=conf['filter_size']))

    batch_size_exp_lower_limit = conf['batch_size_exp_lower_limit']
    batch_size_exp_upper_limit = conf['batch_size_exp_upper_limit']
    learning_rate_lower_limit = conf['learning_rate_lower_limit']
    learning_rate_upper_limit = conf['learning_rate_upper_limit']

    dataset = RadarDataset(hyperparameter_config.input_data_source,
                           hyperparameter_config.mat_path,
                           hyperparameter_config.scaler,
                           is_classification=False)

    hyperparameter_config.input_size = dataset.num_values_per_sample

    # learning rate #
    if learning_rate_lower_limit == learning_rate_upper_limit:
        lr = learning_rate_lower_limit
    else:
        lr = loguniform(learning_rate_lower_limit, learning_rate_upper_limit, 1)[0]
    assert (learning_rate_lower_limit <= lr <= learning_rate_upper_limit)
    hyperparameter_config.learning_rate = lr
    # batch size #
    if batch_size_exp_lower_limit == batch_size_exp_upper_limit:
        batch_size = int(math.pow(2, batch_size_exp_lower_limit))
    else:
        batch_size = int(math.pow(2, int(np.random.randint(batch_size_exp_lower_limit, batch_size_exp_upper_limit, 1))))
    if batch_size > hyperparameter_config.model.max_batch_size:
        batch_size = hyperparameter_config.model.max_batch_size
    hyperparameter_config.batch_size = batch_size

    return dataset, hyperparameter_config
