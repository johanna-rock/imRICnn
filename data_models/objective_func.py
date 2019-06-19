
import torch
from enum import Enum

from sklearn.metrics import mean_squared_error
from torch import nn
from torch.nn.modules.loss import _Loss
import numpy as np
from datasets.radar_dataset import DataContent
from run_scripts import print_, device


class DeltaSNR(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(DeltaSNR, self).__init__(size_average, reduce, reduction)
        self.data_content = DataContent.COMPLEX_PACKET_RD  # extend to others?!

    def forward(self, output_re_im, target_re_im, object_mask, noise_mask):
        object_mask = object_mask.to(device)
        noise_mask = noise_mask.to(device)
        sinr_delta_mean = 0
        num_packets = target_re_im.shape[0]
        if self.data_content is DataContent.COMPLEX_PACKET_RD:
            for p in range(num_packets):
                output_re_im_packet = output_re_im[p]
                target_re_im_packet = target_re_im[p]

                sinr_output = sinr_from_re_im_format(output_re_im_packet, object_mask, noise_mask)
                sinr_target = sinr_from_re_im_format(target_re_im_packet, object_mask, noise_mask)
                sinr_delta_mean += torch.abs(sinr_target - sinr_output)
        else:
            print_('WARNING: Not implemented yet.')
            assert False

        sinr_delta_mean /= num_packets
        return sinr_delta_mean


class SINRLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(SINRLoss, self).__init__(size_average, reduce, reduction)
        self.data_content = DataContent.COMPLEX_PACKET_RD  # extend to others?!

    def forward(self, output_re_im, target_re_im, object_mask, noise_mask):
        object_mask = object_mask.to(device)
        noise_mask = noise_mask.to(device)
        neg_sinr_mean = 0
        num_packets = target_re_im.shape[0]
        if self.data_content is DataContent.COMPLEX_PACKET_RD:
            for p in range(num_packets):
                output_re_im_packet = output_re_im[p]
                neg_sinr_mean -= sinr_from_re_im_format(output_re_im_packet, object_mask, noise_mask)
        else:
            print_('WARNING: Not implemented yet.')
            assert False

        neg_sinr_mean /= num_packets
        return neg_sinr_mean


class MSEWeightedMagPhase(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(MSEWeightedMagPhase, self).__init__(size_average, reduce, reduction)
        self.data_content = DataContent.COMPLEX_PACKET_RD  # extend to others?!
        self.mse = nn.MSELoss()
        self.w_mag = 0.0
        self.w_phase = 0.0
        self.w_re_im = 1.0
        self.epoch = 0

    def forward(self, output_re_im, target_re_im, object_mask, noise_mask):
        object_mask = object_mask.to(device)
        loss = 0
        num_packets = target_re_im.shape[0]
        num_re = int(target_re_im.shape[2] / 2)
        if self.data_content is DataContent.COMPLEX_PACKET_RD:
            for p in range(num_packets):
                output_re_im_packet = output_re_im[p]
                target_re_im_packet = target_re_im[p]

                output_re_packet = output_re_im_packet[:, :num_re]
                output_im_packet = output_re_im_packet[:, num_re:]

                target_re_packet = target_re_im_packet[:, :num_re]
                target_im_packet = target_re_im_packet[:, num_re:]

                output_peaks_re = torch.masked_select(output_re_packet, object_mask)
                output_peaks_im = torch.masked_select(output_im_packet, object_mask)
                target_peaks_re = torch.masked_select(target_re_packet, object_mask)
                target_peaks_im = torch.masked_select(target_im_packet, object_mask)
                phase_target = torch.atan(target_peaks_im / target_peaks_re)
                phase_output = torch.atan(output_peaks_im / output_peaks_re)

                target_max_mag = torch.sqrt(target_re_packet ** 2 + target_im_packet ** 2).view(-1).max()
                target_re_packet_log_mag = target_re_packet / target_max_mag
                target_im_packet_log_mag = target_im_packet / target_max_mag
                target_log_mag = 10 * torch.log10(torch.sqrt(target_re_packet_log_mag ** 2 + target_im_packet_log_mag ** 2))
                target_log_mag = torch.masked_select(target_log_mag, object_mask)

                output_max_mag = torch.sqrt(output_re_packet ** 2 + output_im_packet ** 2).view(-1).max()
                output_re_packet_log_mag = output_re_packet / output_max_mag
                output_im_packet_log_mag = output_im_packet / output_max_mag
                output_log_mag = 10 * torch.log10(torch.sqrt(output_re_packet_log_mag ** 2 + output_im_packet_log_mag ** 2))
                output_log_mag = torch.masked_select(output_log_mag, object_mask)

                loss += self.w_re_im * self.mse(output_re_im, target_re_im) +\
                        self.w_mag * self.mse(output_log_mag, target_log_mag) +\
                        self.w_phase * self.mse(phase_output, phase_target)
        else:
            print_('WARNING: Not implemented yet.')
            assert False

        loss /= num_packets
        return loss

    def next_epoch(self):
        pass
        self.epoch += 1
        if self.epoch % 10 == 0 and self.w_re_im > 0.4:
            self.w_re_im -= 0.1
            self.w_mag = (1 - self.w_re_im) / 2
            self.w_phase = (1 - self.w_re_im) / 2


class MSE(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(MSE, self).__init__(size_average, reduce, reduction)
        self.mse = nn.MSELoss()

    def forward(self, output_re_im, target_re_im, object_mask, noise_mask):
        return self.mse.forward(output_re_im, target_re_im)


class ObjectiveFunction(Enum):
    DELTA_SNR = DeltaSNR()
    MSE = MSE()
    MSE_MAG_PHASE_WEIGHTED = MSEWeightedMagPhase()
    SINR = SINRLoss()

    def __call__(self, *args):
        return self.value(*args)

    @staticmethod
    def loss_to_running_loss(batch_loss, batch_size):
        return batch_loss * batch_size

    @staticmethod
    def loss_from_running_loss(running_loss, sample_size):
        return running_loss / sample_size

    @staticmethod
    def from_name(value):
        if value == ObjectiveFunction.DELTA_SNR.name:
            return ObjectiveFunction.DELTA_SNR
        elif value == ObjectiveFunction.MSE.name:
            return ObjectiveFunction.MSE
        elif value == ObjectiveFunction.MSE_MAG_PHASE_WEIGHTED.name:
            return ObjectiveFunction.MSE_MAG_PHASE_WEIGHTED
        elif value == ObjectiveFunction.SINR.name:
            return ObjectiveFunction.SINR
        else:
            return None

    @staticmethod
    def objective_func_name(func):
        try:
            if func.name in ObjectiveFunction.__members__:
                return func.name
            else:
                return 'None'
        except AttributeError:
            return 'None'


def sinr_log_mag(log_mag_rd_target, log_mag_rd_test, object_mask, noise_mask):
    return np.average(log_mag_rd_test[object_mask]) - np.average(log_mag_rd_test[noise_mask])


def sinr(rd_target, rd_test, object_mask, noise_mask):

    rd_test_mag = np.abs(rd_test)**2

    obj_values = rd_test_mag[object_mask]
    obj_magnitude = np.average(obj_values)

    noise_values = rd_test_mag[noise_mask]
    noise_magnitude = np.average(noise_values)

    return 10 * np.log10(obj_magnitude / noise_magnitude)


def sinr_1d(cr_target, cr_test, object_mask, noise_mask):

    cr_test_mag = np.abs(cr_test)**2

    obj_values = cr_test_mag[object_mask]
    obj_magnitude = np.average(obj_values)

    noise_values = cr_test_mag[noise_mask]
    noise_magnitude = np.average(noise_values)

    return 10 * np.log10(obj_magnitude / noise_magnitude)


def sinr_from_re_im_format(re_im_packet, obj_mask, noise_mask):
    if len(re_im_packet.shape) == 3:
        re_im_packet = re_im_packet[0]

    num_re = int(re_im_packet.shape[1]/2)
    re_packet = re_im_packet[:, :num_re]
    im_packet = re_im_packet[:, num_re:]

    mag = re_packet ** 2 + im_packet ** 2

    obj_values = torch.masked_select(mag, obj_mask)
    obj_magnitude = torch.mean(obj_values)

    noise_values = torch.masked_select(mag, noise_mask)
    noise_magnitude = torch.mean(noise_values)

    return 10 * torch.log10(obj_magnitude / noise_magnitude)


def peak_mag_mse(log_mag_rd_target, log_mag_rd_test, object_mask, noise_mask):

    obj_values_target = log_mag_rd_target[object_mask]
    obj_values_test = log_mag_rd_test[object_mask]

    if len(obj_values_target) == 0:
        return np.nan
    return mean_squared_error(obj_values_target, obj_values_test)


def evm(rd_target, rd_test, object_mask, noise_mask):

    obj_values_target = rd_target[object_mask]
    obj_values_test = rd_test[object_mask]

    if len(obj_values_target) == 0:
        return np.nan
    evms = np.abs(obj_values_target - obj_values_test) / np.abs(obj_values_target)
    return np.average(evms)


def evm_norm(rd_target, rd_test, object_mask, noise_mask):
    rd_target_norm = rd_target / np.amax(np.abs(rd_target))
    rd_test_norm = rd_test / np.amax(np.abs(rd_test))

    obj_values_target = rd_target_norm[object_mask]
    obj_values_test = rd_test_norm[object_mask]

    if len(obj_values_target) == 0:
        return np.nan
    evms = np.abs(obj_values_target - obj_values_test) / np.abs(obj_values_target)
    return np.average(evms)


def evm_1d(cr_target, cr_test, object_mask, noise_mask):
    obj_values_target = cr_target[object_mask]
    obj_values_test = cr_test[object_mask]

    if len(obj_values_target) == 0:
        return np.nan
    evms = np.abs(obj_values_target - obj_values_test) / np.abs(obj_values_target)
    return np.average(evms)


def evm_1d_norm(cr_target, cr_test, object_mask, noise_mask):
    cr_target_norm = cr_target / np.amax(np.abs(cr_target))
    cr_test_norm = cr_test / np.amax(np.abs(cr_test))

    obj_values_target = cr_target_norm[object_mask]
    obj_values_test = cr_test_norm[object_mask]

    if len(obj_values_target) == 0:
        print_('WARNING: no obj peak targets found in evm_1d_norm!')
        return np.nan
    evms = np.abs(obj_values_target - obj_values_test) / np.abs(obj_values_target)
    return np.average(evms)


def rd_obj_peak_phase_mse(rd_target, rd_test, object_mask, noise_mask):

    peaks_target = rd_target[object_mask]
    peaks_test = rd_test[object_mask]

    if len(peaks_target) == 0:
        print_('WARNING: no peaks found for evaluation metric.')
        return np.nan

    peaks_target_imag = np.imag(peaks_target)
    peaks_target_real = np.real(peaks_target)
    peaks_phase_target = np.arctan(peaks_target_imag.astype('float') / peaks_target_real.astype('float'))

    peaks_test_imag = np.imag(peaks_test)
    peaks_test_real = np.real(peaks_test)
    peaks_phase_test = np.arctan(peaks_test_imag.astype('float') / peaks_test_real.astype('float'))

    phase_mse = mean_squared_error(peaks_phase_target, peaks_phase_test)
    return phase_mse


def rd_obj_peak_log_mag_mse(rd_target, rd_test, object_mask, noise_mask):

    peaks_target = rd_target[object_mask]
    peaks_test = rd_test[object_mask]

    if len(peaks_target) == 0:
        print_('WARNING: no peaks found for evaluation metric.')
        return np.nan

    mag_target = np.abs(peaks_target)
    mag_test = np.abs(peaks_test)

    phase_mse = mean_squared_error(mag_target, mag_test)
    return phase_mse
