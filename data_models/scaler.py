import functools
from enum import Enum
import numpy as np


class ZeroMeanUnitVarianceScaler():
    def __init__(self):
        self.mean = None
        self.var = None

    def transform(self, data):
        if self.mean is not None and self.var is not None and np.any(np.iscomplex(data)):
            mean_re_im = (np.real(self.mean) + np.imag(self.mean)) / 2
            data = data - (mean_re_im + 1j * mean_re_im)
            data = data / self.var

        return data

    def inverse_transform(self, data):
        if self.mean is not None and self.var is not None and np.any(np.iscomplex(data)):
            mean_re_im = (np.real(self.mean) + np.imag(self.mean)) / 2
            data = data * self.var
            data = data + (mean_re_im + 1j * mean_re_im)

        return data


class ComplexFeatureScaler():
    def __init__(self):
        self.mean_complex = None
        self.sr_cov = None
        self.inv_sr_cov = None

    def transform(self, data):
        assert self.mean_complex is not None and self.inv_sr_cov is not None

        data = data - self.mean_complex
        d_real = np.real(data) * self.inv_sr_cov[0, 0] + np.imag(data) * self.inv_sr_cov[0, 1]
        d_imag = np.real(data) * self.inv_sr_cov[1, 0] + np.imag(data) * self.inv_sr_cov[1, 1]
        return d_real + 1j * d_imag

    def inverse_transform(self, data):
        assert self.mean_complex is not None and self.sr_cov is not None
        d_real = np.real(data) * self.sr_cov[0, 0] + np.imag(data) * self.sr_cov[0, 1]
        d_imag = np.real(data) * self.sr_cov[1, 0] + np.imag(data) * self.sr_cov[1, 1]
        data = d_real + 1j * d_imag
        return data + self.mean_complex


class Scaler(Enum):
    STD_SCALER = functools.partial(ZeroMeanUnitVarianceScaler)
    COMPLEX_FEATURE_SCALER = functools.partial(ComplexFeatureScaler)

    def __call__(self, *args):
        return self.value(*args)

    @staticmethod
    def from_name(value):
        if value == Scaler.STD_SCALER.name:
            return Scaler.STD_SCALER
        elif value == Scaler.COMPLEX_FEATURE_SCALER.name:
            return Scaler.COMPLEX_FEATURE_SCALER
        else:
            return None

    @staticmethod
    def scaler_name(scaler):
        try:
            if scaler.name in Scaler.__members__:
                return scaler.name
            else:
                return 'None'
        except AttributeError:
            return 'None'
