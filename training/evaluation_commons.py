from enum import Enum

from data_models.objective_func import sinr, evm, sinr_1d, evm_1d, rd_obj_peak_phase_mse, \
    rd_obj_peak_log_mag_mse, evm_1d_norm, evm_norm, peak_mag_mse, sinr_log_mag


class EvaluationFunction(Enum):
    SINR_RD = 0
    EVM_RD = 1
    SINR_CR = 5
    EVM_CR = 6
    PHASE_MSE_RD = 7
    LOG_MAG_MSE_RD = 8
    EVM_RD_NORM = 9
    EVM_CR_NORM = 10
    PEAK_MAG_MSE = 11
    SINR_RD_LOG_MAG = 12

    def __call__(self, *args):
        return self.func()(*args)

    def label(self):
        if self is EvaluationFunction.SINR_RD:
            return 'rd-sinr'
        elif self is EvaluationFunction.EVM_RD:
            return 'rd-evm'
        elif self is EvaluationFunction.SINR_CR:
            return 'cr-sinr'
        elif self is EvaluationFunction.EVM_CR:
            return 'cr-evm'
        elif self is EvaluationFunction.PHASE_MSE_RD:
            return 'rd-phase-mse'
        elif self is EvaluationFunction.LOG_MAG_MSE_RD:
            return 'rd-log-mag-mse'
        elif self is EvaluationFunction.EVM_RD_NORM:
            return 'rd-evm-norm'
        elif self is EvaluationFunction.EVM_CR_NORM:
            return 'cr-evm-norm'
        elif self is EvaluationFunction.PEAK_MAG_MSE:
            return 'peak-mag-mse'
        elif self is EvaluationFunction.SINR_RD_LOG_MAG:
            return 'lm-rd-sinr'

    def func(self):
        if self is EvaluationFunction.SINR_RD:
            return sinr
        elif self is EvaluationFunction.EVM_RD:
            return evm
        elif self is EvaluationFunction.SINR_CR:
            return sinr_1d
        elif self is EvaluationFunction.EVM_CR:
            return evm_1d
        elif self is EvaluationFunction.PHASE_MSE_RD:
            return rd_obj_peak_phase_mse
        elif self is EvaluationFunction.LOG_MAG_MSE_RD:
            return rd_obj_peak_log_mag_mse
        elif self is EvaluationFunction.EVM_RD_NORM:
            return evm_norm
        elif self is EvaluationFunction.EVM_CR_NORM:
            return evm_1d_norm
        elif self is EvaluationFunction.PEAK_MAG_MSE:
            return peak_mag_mse
        elif self is EvaluationFunction.SINR_RD_LOG_MAG:
            return sinr_log_mag


class Signal(Enum):
    PREDICTION = 0
    PREDICTION_SUBSTITUDE = 1
    CLEAN = 2
    INTERFERED = 3
    CLEAN_NOISE = 4
    BASELINE_ZERO_SUB = 5

    def label(self):
        if self is Signal.PREDICTION:
            return 'prediction'
        elif self is Signal.PREDICTION_SUBSTITUDE:
            return 'pred substi'
        elif self is Signal.CLEAN:
            return 'clean'
        elif self is Signal.INTERFERED:
            return 'interfered'
        elif self is Signal.CLEAN_NOISE:
            return 'clean + noise'
        elif self is Signal.BASELINE_ZERO_SUB:
            return 'miti: 0 sub'
