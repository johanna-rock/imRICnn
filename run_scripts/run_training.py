import torch

from data_models.objective_func import ObjectiveFunction
from data_models.scaler import Scaler
from datasets.radar_dataset import DataSource
from models.ri_cnn_rd import RICNN_RD, MAG_CNN_RD
from models.ri_cnn_rp import RICNN_RP
from run_scripts import JOB_DIR, task_id
from training.sample_hyperparameters import select_and_sample_hyperparameter_config_for_cnn
from training.trainer import train_with_hyperparameter_config


def run_signal_denoising_cnn_re_im():
    task = 'Denoising'

    # DATA SOURCE
    # DataSource.DENOISE_REAL_IMAG_RD
    # DataSource.DENOISE_LOG_MAG_RD
    # DataSource.DENOISE_REAL_IMAG_RAMP  # Note: data set must contain RP sampels
    datasource = DataSource.DENOISE_REAL_IMAG_RD

    # SCALER
    # Scaler.STD_SCALER
    # Scaler.COMPLEX_FEATURE_SCALER2
    scaler = Scaler.STD_SCALER

    # CRITERION
    # ObjectiveFunction.MSE
    # ObjectiveFunction.SINR
    # ObjectiveFunction.MSE_MAG_PHASE_WEIGHTED
    criterion = ObjectiveFunction.MSE

    # MODELS
    # RICNN_RP
    # RICNN_RD
    # MAG_CNN_RD

    if datasource is DataSource.DENOISE_LOG_MAG_RD:
        model = MAG_CNN_RD
    elif datasource is DataSource.DENOISE_REAL_IMAG_RD:
        model = RICNN_RD
    elif datasource is DataSource.DENOISE_REAL_IMAG_RAMP:
        model = RICNN_RP
    else:
        assert False

    configurations = [
        {'data_source': datasource, 'model': model,
         'mat_path': "sim_200x1+25x8+25x8_1-3i", 'scaler': scaler, 'criterion': criterion,
         'num_conv_layer': 4, 'num_filters': 16, 'filter_size': (3, 3),
         'num_epochs': 1}
    ]

    # default parameters
    for c in configurations:
        try:
            c['num_conv_layer']
        except KeyError:
            c['num_conv_layer'] = None

        try:
            c['num_filters']
        except KeyError:
            c['num_filters'] = None

        try:
            c['filter_size']
        except KeyError:
            c['filter_size'] = None

        try:
            c['num_epochs']
        except KeyError:
            c['num_epochs'] = 100

        c['learning_rate_lower_limit'] = 0.00005
        c['learning_rate_upper_limit'] = 0.00005
        c['batch_size_exp_lower_limit'] = 1
        c['batch_size_exp_upper_limit'] = 1

        # c['learning_rate_lower_limit'] = 0.000005
        # c['learning_rate_upper_limit'] = 0.05
        # c['batch_size_exp_lower_limit'] = 1
        # c['batch_size_exp_upper_limit'] = 10

    dataset, hyperparameter_config = select_and_sample_hyperparameter_config_for_cnn(configurations)
    model, hyperparameters, evaluation_result = train_with_hyperparameter_config(dataset, hyperparameter_config, task)

    model = model.to(torch.device('cpu'))
    model_path = '{}/model_id{}'.format(JOB_DIR, task_id)
    torch.save(model, model_path)


run_signal_denoising_cnn_re_im()
