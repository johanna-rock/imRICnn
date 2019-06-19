import warnings
from enum import Enum
import torch
from data_models.scaler import Scaler
from datasets.radar_dataset import RadarDataset, DatasetPartition, DataContent, DataSource
from run_scripts import REPO_BASE_DIR, device
from training.rd_evaluation import evaluate_rd


class PretrainedModels(Enum):
    MODEL_A = 0
    MODEL_D = 1

    @staticmethod
    def model_path(model):
        if model == PretrainedModels.MODEL_A:
            return "modelA"
        elif model == PretrainedModels.MODEL_D:
            return "modelD"


def run_evaluation():
    data_source = DataSource.DENOISE_REAL_IMAG_RD
    mat_path = 'sim_200x1+25x8+25x8_1-3i'
    scaler = Scaler.COMPLEX_FEATURE_SCALER
    model = PretrainedModels.MODEL_D  # choose pre-trained model {PretrainedModels.MODEL_A, PretrainedModels.MODEL_D}

    dataset = RadarDataset(data_source, mat_path, scaler)
    test_dataset = dataset.clone_for_new_active_partition(DatasetPartition.TEST)

    if len(test_dataset) <= 0:
        warnings.warn('Test data set empty.')
        return

    try:
        model = torch.load(REPO_BASE_DIR + "/results/trained_models/" + PretrainedModels.model_path(model)).to(device)
    except FileNotFoundError:
        warnings.warn('Model not found.')
        return

    if dataset.data_content is DataContent.COMPLEX_PACKET_RD or dataset.data_content is DataContent.COMPLEX_RAMP:
        evaluate_rd(model, test_dataset, 'evaluation_test_rd')


run_evaluation()
