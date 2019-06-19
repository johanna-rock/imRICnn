import copy
import os
import scipy
import warnings
from enum import Enum
import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.io as spio
from data_models.scaler import Scaler
from run_scripts import print_
from utils.rd_processing import calculate_velocity_fft, calculate_angle_fft, v_vec_fft2, d_vec_fft2, num_angle_fft_bins, \
    d_max


def split_indices_for_partitions(num_items, train_ratio=0.5, val_ratio=0.5, test_ratio=0.0):
    assert(train_ratio + val_ratio + test_ratio == 1.0)

    train_size = int(num_items * train_ratio)
    val_size = int(num_items * val_ratio)

    indices = list(range(num_items))
    train_indices = indices[0:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    return train_indices, val_indices, test_indices


def load_data_for_denoising_ri_ramps(out, dataset):
    try:  # for real measurements
        measurements = out['test_rd'][()].transpose()

        # second FFT
        num_ramps = measurements.shape[0]
        num_fts = measurements.shape[1]
        assert (num_ramps % dataset.num_ramps_per_packet == 0)

        x = measurements.reshape(num_ramps, 1, num_fts)

        filter_mask = np.ones((x.shape[0],), dtype=int)
        interference_mask = filter_mask
        all_noise_mask = np.ones(measurements.shape, dtype=bool)

        return x, x, x, x, x, filter_mask, interference_mask, x,\
               all_noise_mask, all_noise_mask, all_noise_mask, all_noise_mask, []
    except ValueError:
        pass

    fft_original = out['s_IF_clean_noise'][()].transpose()  # IF clean + IF gaussian noise
    fft_interf = out['s_IF'][()].transpose()  # IF clean + IF gaussian noise + IF interference
    fft_clean = out['s_IF_clean'][()].transpose()  # IF clean
    interference_mask = out['interference_active_ramp'][()].transpose()
    fft_zero_mitigation = out['s_IF_zero_interf_td'][()].transpose()
    object_targets = out['objects'][()]

    num_ramps = fft_clean.shape[0]
    num_fts = fft_clean.shape[1]
    num_packets = int(num_ramps / dataset.num_ramps_per_packet)
    x = fft_interf.reshape(num_ramps, 1, num_fts)
    y = fft_clean.reshape(num_ramps, 1, num_fts)

    rd_object_masks = []
    aoa_object_masks = []
    rd_noise_masks = []
    aoa_noise_masks = []
    target_angles = []
    for p in range(num_packets):

        target_ranges = np.array([object_targets[p][4]]).flatten()
        target_velocities = np.array([object_targets[p][7]]).flatten()
        ta = np.array([object_targets[p][5]]).flatten()

        rd_o_masks, rd_n_masks, d_indices, v_indices = calculate_rd_object_and_noise_masks(target_ranges, target_velocities, num_fts, dataset.num_ramps_per_packet)
        aoa_o_masks, aoa_n_masks = calculate_aoa_object_and_noise_masks(target_ranges, ta, num_fts, num_angle_fft_bins)
        target_angles.append({'d': d_indices, 'v': v_indices, 'a': ta})

        rd_object_masks.append(rd_o_masks * dataset.num_ramps_per_packet)
        rd_noise_masks.append(rd_n_masks * dataset.num_ramps_per_packet)

        aoa_object_masks.append(aoa_o_masks * dataset.num_ramps_per_packet)
        aoa_noise_masks.append(aoa_n_masks * dataset.num_ramps_per_packet)

    rd_object_masks = np.array(rd_object_masks)
    rd_noise_masks = np.array(rd_noise_masks)

    aoa_object_masks = np.array(aoa_object_masks)
    aoa_noise_masks = np.array(aoa_noise_masks)

    filter_mask = np.ones((x.shape[0],), dtype=int)

    return x, y, fft_clean, fft_original, fft_interf, filter_mask, interference_mask, fft_zero_mitigation,\
           rd_object_masks, rd_noise_masks, aoa_object_masks, aoa_noise_masks, target_angles


def load_data_for_denoising_ri_ramps_training_with_interfered_ramps_only(out, dataset):
    try:  # for real measurements
        measurements = out['test_rd'][()].transpose()

        # second FFT
        num_ramps = measurements.shape[0]
        num_fts = measurements.shape[1]
        assert (num_ramps % dataset.num_ramps_per_packet == 0)

        x = measurements.reshape(num_ramps, 1, num_fts)

        filter_mask = np.ones((x.shape[0],), dtype=int)
        interference_mask = filter_mask
        all_noise_mask = np.ones(measurements.shape, dtype=bool)

        return x, x, x, x, x, filter_mask, interference_mask, x,\
               all_noise_mask, all_noise_mask, all_noise_mask, all_noise_mask, []
    except ValueError:
        pass

    fft_original = out['s_IF_clean_noise'][()].transpose()  # IF clean + IF gaussian noise
    fft_interf = out['s_IF'][()].transpose()  # IF clean + IF gaussian noise + IF interference
    fft_clean = out['s_IF_clean'][()].transpose()  # IF clean
    filter_mask = out['interference_active_ramp'][()].transpose()
    fft_zero_mitigation = out['s_IF_zero_interf_td'][()].transpose()
    interference_mask = filter_mask
    object_targets = out['objects'][()]

    num_ramps = fft_clean.shape[0]
    num_fts = fft_clean.shape[1]
    num_packets = int(num_ramps / dataset.num_ramps_per_packet)
    x = fft_interf.reshape(num_ramps, 1, num_fts)
    y = fft_clean.reshape(num_ramps, 1, num_fts)

    rd_object_masks = []
    aoa_object_masks = []
    rd_noise_masks = []
    aoa_noise_masks = []
    target_angles = []
    for p in range(num_packets):

        target_ranges = np.array([object_targets[p][4]]).flatten()
        target_velocities = np.array([object_targets[p][7]]).flatten()
        ta = np.array([object_targets[p][5]]).flatten()

        rd_o_masks, rd_n_masks, d_indices, v_indices = calculate_rd_object_and_noise_masks(target_ranges, target_velocities, num_fts, dataset.num_ramps_per_packet)
        aoa_o_masks, aoa_n_masks = calculate_aoa_object_and_noise_masks(target_ranges, ta, num_fts, num_angle_fft_bins)
        target_angles.append({'d': d_indices, 'v': v_indices, 'a': ta})

        rd_object_masks.append(rd_o_masks * dataset.num_ramps_per_packet)
        rd_noise_masks.append(rd_n_masks * dataset.num_ramps_per_packet)

        aoa_object_masks.append(aoa_o_masks * dataset.num_ramps_per_packet)
        aoa_noise_masks.append(aoa_n_masks * dataset.num_ramps_per_packet)

    rd_object_masks = np.array(rd_object_masks)
    rd_noise_masks = np.array(rd_noise_masks)

    aoa_object_masks = np.array(aoa_object_masks)
    aoa_noise_masks = np.array(aoa_noise_masks)

    return x, y, fft_clean, fft_original, fft_interf, filter_mask, interference_mask, fft_zero_mitigation,\
           rd_object_masks, rd_noise_masks, aoa_object_masks, aoa_noise_masks, []


def load_data_for_denoising_ri_range_doppler_map(out, dataset):
    try:  # for real measurements
        measurements = out['test_rd'][()].transpose()

        # second FFT
        num_ramps = measurements.shape[0]
        assert (num_ramps % dataset.num_ramps_per_packet == 0)
        num_packets = int(num_ramps / dataset.num_ramps_per_packet)

        rd = []
        for p in range(num_packets):
            rd.append(calculate_velocity_fft(
                measurements[p * dataset.num_ramps_per_packet: (p + 1) * dataset.num_ramps_per_packet]))

        rd = np.array(rd)

        filter_mask = np.ones((rd.shape[0],), dtype=int)
        interference_mask = filter_mask
        all_noise_mask = np.ones(measurements.shape, dtype=bool)

        return rd, rd, rd, rd, rd, filter_mask, interference_mask, rd,\
               all_noise_mask, all_noise_mask, all_noise_mask, all_noise_mask, []
    except ValueError:
        pass

    fft_original = out['s_IF_clean_noise'][()].transpose()  # IF clean + IF gaussian noise
    fft_interf = out['s_IF'][()].transpose()  # IF clean + IF gaussian noise + IF interference
    fft_clean = out['s_IF_clean'][()].transpose()  # IF clean
    fft_zero_mitigation = out['s_IF_zero_interf_td'][()].transpose()
    object_targets = out['objects'][()]

    # second FFT
    num_ramps = fft_original.shape[0]
    num_fts = fft_original.shape[1]
    assert (num_ramps % dataset.num_ramps_per_packet == 0)
    num_packets = int(num_ramps / dataset.num_ramps_per_packet)

    rd_original = []
    rd_interf = []
    rd_clean = []
    rd_zero_mitigation = []
    rd_object_masks = []
    aoa_object_masks = []
    # cr_object_masks = []
    rd_noise_masks = []
    aoa_noise_masks = []
    # cr_noise_masks = []
    target_angles = []
    for p in range(num_packets):
        rd_original.append(calculate_velocity_fft(fft_original[p*dataset.num_ramps_per_packet: (p+1)*dataset.num_ramps_per_packet]))
        rd_interf.append(calculate_velocity_fft(
            fft_interf[p * dataset.num_ramps_per_packet: (p + 1) * dataset.num_ramps_per_packet]))

        rd_clean.append(calculate_velocity_fft(
            fft_clean[p * dataset.num_ramps_per_packet: (p + 1) * dataset.num_ramps_per_packet]))
        rd_zero_mitigation.append(calculate_velocity_fft(
            fft_zero_mitigation[p * dataset.num_ramps_per_packet: (p + 1) * dataset.num_ramps_per_packet]))

        target_ranges = np.array([object_targets[p][4]]).flatten()
        target_velocities = np.array([object_targets[p][7]]).flatten()
        ta = np.array([object_targets[p][5]]).flatten()

        rd_o_masks, rd_n_masks, d_indices, v_indices = calculate_rd_object_and_noise_masks(target_ranges, target_velocities, num_fts, dataset.num_ramps_per_packet)
        aoa_o_masks, aoa_n_masks = calculate_aoa_object_and_noise_masks(target_ranges, ta, num_fts, num_angle_fft_bins)
        target_angles.append({'d': d_indices, 'v': v_indices, 'a': ta})

        rd_object_masks.append(rd_o_masks)
        rd_noise_masks.append(rd_n_masks)

        aoa_object_masks.append(aoa_o_masks)
        aoa_noise_masks.append(aoa_n_masks)

    rd_object_masks = np.array(rd_object_masks)
    rd_noise_masks = np.array(rd_noise_masks)

    aoa_object_masks = np.array(aoa_object_masks)
    aoa_noise_masks = np.array(aoa_noise_masks)

    rd_original = np.array(rd_original)
    rd_interf = np.array(rd_interf)
    rd_clean = np.array(rd_clean)
    rd_zero_mitigation = np.array(rd_zero_mitigation)

    x = rd_interf
    y = rd_clean  # y = rd_original

    filter_mask = np.ones((x.shape[0],), dtype=int)
    interference_mask = filter_mask

    return x, y, rd_clean, rd_original, rd_interf, filter_mask, interference_mask, rd_zero_mitigation,\
           rd_object_masks, rd_noise_masks, aoa_object_masks, aoa_noise_masks, target_angles


def load_data_for_denoising_log_mag_range_doppler_map(out, dataset):
    try:  # for real measurements
        measurements = out['test_rd'][()].transpose()

        # second FFT
        num_ramps = measurements.shape[0]
        assert (num_ramps % dataset.num_ramps_per_packet == 0)
        num_packets = int(num_ramps / dataset.num_ramps_per_packet)

        rd = []
        for p in range(num_packets):
            fft2 = calculate_velocity_fft(
                measurements[p * dataset.num_ramps_per_packet: (p + 1) * dataset.num_ramps_per_packet])
            fft2 = fft2 / np.amax(np.abs(fft2))
            fft2 = 10 * np.log10(np.abs(fft2)**2)
            rd.append(fft2)

        rd = np.array(rd)

        filter_mask = np.ones((rd.shape[0],), dtype=int)
        interference_mask = filter_mask
        all_noise_mask = np.ones(measurements.shape, dtype=bool)

        return rd, rd, rd, rd, rd, filter_mask, interference_mask, rd,\
               all_noise_mask, all_noise_mask, all_noise_mask, all_noise_mask, []
    except ValueError:
        pass

    fft_original = out['s_IF_clean_noise'][()].transpose()  # IF clean + IF gaussian noise
    fft_interf = out['s_IF'][()].transpose()  # IF clean + IF gaussian noise + IF interference
    fft_clean = out['s_IF_clean'][()].transpose()  # IF clean
    fft_zero_mitigation = out['s_IF_zero_interf_td'][()].transpose()
    object_targets = out['objects'][()]

    # second FFT
    num_ramps = fft_original.shape[0]
    num_fts = fft_original.shape[1]
    assert (num_ramps % dataset.num_ramps_per_packet == 0)
    num_packets = int(num_ramps / dataset.num_ramps_per_packet)

    rd_original = []
    rd_interf = []
    rd_clean = []
    rd_zero_mitigation = []
    rd_object_masks = []
    aoa_object_masks = []
    rd_noise_masks = []
    aoa_noise_masks = []
    target_angles = []
    for p in range(num_packets):
        fft2 = calculate_velocity_fft(
            fft_original[p * dataset.num_ramps_per_packet: (p + 1) * dataset.num_ramps_per_packet])
        fft2 = fft2 / np.amax(np.abs(fft2))
        fft2 = 10 * np.log10(np.abs(fft2)**2)
        rd_original.append(fft2)

        fft2 = calculate_velocity_fft(
            fft_interf[p * dataset.num_ramps_per_packet: (p + 1) * dataset.num_ramps_per_packet])
        fft2 = fft2 / np.amax(np.abs(fft2))
        fft2 = 10 * np.log10(np.abs(fft2)**2)
        rd_interf.append(fft2)

        fft2 = calculate_velocity_fft(
            fft_clean[p * dataset.num_ramps_per_packet: (p + 1) * dataset.num_ramps_per_packet])
        fft2 = fft2 / np.amax(np.abs(fft2))
        fft2 = 10 * np.log10(np.abs(fft2)**2)
        rd_clean.append(fft2)

        fft2 = calculate_velocity_fft(
            fft_zero_mitigation[p * dataset.num_ramps_per_packet: (p + 1) * dataset.num_ramps_per_packet])
        fft2 = fft2 / np.amax(np.abs(fft2))
        fft2 = 10 * np.log10(np.abs(fft2)**2)
        rd_zero_mitigation.append(fft2)

        target_ranges = np.array([object_targets[p][4]]).flatten()
        target_velocities = np.array([object_targets[p][7]]).flatten()
        target_angles.append(np.array([object_targets[p][5]]).flatten())

        ta = np.array([object_targets[p][5]]).flatten()

        rd_o_masks, rd_n_masks, d_indices, v_indices = calculate_rd_object_and_noise_masks(target_ranges, target_velocities, num_fts, dataset.num_ramps_per_packet)
        aoa_o_masks, aoa_n_masks = calculate_aoa_object_and_noise_masks(target_ranges, ta, num_fts, num_angle_fft_bins)
        target_angles.append({'d': d_indices, 'v': v_indices, 'a': ta})

        rd_object_masks.append(rd_o_masks)
        rd_noise_masks.append(rd_n_masks)

        aoa_object_masks.append(aoa_o_masks)
        aoa_noise_masks.append(aoa_n_masks)

    rd_original = np.array(rd_original)
    rd_interf = np.array(rd_interf)
    rd_clean = np.array(rd_clean)
    rd_zero_mitigation = np.array(rd_zero_mitigation)

    rd_object_masks = np.array(rd_object_masks)
    rd_noise_masks = np.array(rd_noise_masks)

    aoa_object_masks = np.array(aoa_object_masks)
    aoa_noise_masks = np.array(aoa_noise_masks)

    x = rd_interf
    y = rd_clean

    filter_mask = np.ones((x.shape[0],), dtype=int)
    interference_mask = filter_mask

    return x, y, rd_clean, rd_original, rd_interf, filter_mask, interference_mask, rd_zero_mitigation,\
           rd_object_masks, rd_noise_masks, aoa_object_masks, aoa_noise_masks, target_angles


def load_data_for_denoising_ri_angle_map(out, dataset):
    num_channels = dataset.get_num_channels()

    try:  # for real measurements
        measurements = out['test_rd'][()].transpose()

        num_ramps = measurements.shape[0]
        num_fts = measurements.shape[1]
        assert (num_ramps % dataset.num_ramps_per_packet == 0)
        num_packets = int(num_ramps / dataset.num_ramps_per_packet)
        assert (num_packets % num_channels == 0)

        aoa_maps = []
        aoa_map = np.zeros((num_fts, num_channels), dtype=np.complex128)
        for p in range(num_packets):
            c = p % num_channels
            packet_data = measurements[p * dataset.num_ramps_per_packet: (p + 1) * dataset.num_ramps_per_packet]
            aoa_map[:, c] = packet_data[0, :].transpose()
            if c == num_channels-1:
                aoa_maps.append(calculate_angle_fft(aoa_map))

        aoa_maps = np.array(aoa_maps)

        filter_mask = np.ones((aoa_maps.shape[0],), dtype=int)
        interference_mask = filter_mask
        all_noise_mask = np.ones(measurements.shape, dtype=bool)

        return aoa_maps, aoa_maps, aoa_maps, aoa_maps, aoa_maps, filter_mask, interference_mask, aoa_maps,\
               all_noise_mask, all_noise_mask, all_noise_mask, all_noise_mask, []
    except ValueError:
        pass

    fft_original = out['s_IF_clean_noise'][()].transpose()  # IF clean + IF gaussian noise
    fft_interf = out['s_IF'][()].transpose()  # IF clean + IF gaussian noise + IF interference
    fft_clean = out['s_IF_clean'][()].transpose()  # IF clean
    fft_zero_mitigation = out['s_IF_zero_interf_td'][()].transpose()
    object_targets = out['objects'][()]

    # angle FFT
    num_ramps = fft_original.shape[0]
    num_fts = fft_original.shape[1]
    num_packets = int(num_ramps / dataset.num_ramps_per_packet)
    assert (num_ramps % dataset.num_ramps_per_packet == 0)
    assert (num_packets % num_channels == 0)

    aoa_original = []
    aoa_interf = []
    aoa_clean = []
    aoa_zero_mitigation = []

    aoa_map_original = np.zeros((num_fts, num_channels), dtype=np.complex128)
    aoa_map_interf = np.zeros((num_fts, num_channels), dtype=np.complex128)
    aoa_map_clean = np.zeros((num_fts, num_channels), dtype=np.complex128)
    aoa_map_zero_mitigation = np.zeros((num_fts, num_channels), dtype=np.complex128)

    rd_object_masks = []
    aoa_object_masks = []
    rd_noise_masks = []
    aoa_noise_masks = []
    target_angles = []

    for p in range(num_packets):
        c = p % num_channels

        packet_data = fft_original[p * dataset.num_ramps_per_packet: (p + 1) * dataset.num_ramps_per_packet]
        aoa_map_original[:, c] = packet_data[0, :]

        packet_data = fft_interf[p * dataset.num_ramps_per_packet: (p + 1) * dataset.num_ramps_per_packet]
        aoa_map_interf[:, c] = packet_data[0, :].transpose()

        packet_data = fft_clean[p * dataset.num_ramps_per_packet: (p + 1) * dataset.num_ramps_per_packet]
        aoa_map_clean[:, c] = packet_data[0, :]

        packet_data = fft_zero_mitigation[p * dataset.num_ramps_per_packet: (p + 1) * dataset.num_ramps_per_packet]
        aoa_map_zero_mitigation[:, c] = packet_data[0, :]

        if c == num_channels - 1:
            aoa_original.append(calculate_angle_fft(aoa_map_original))
            aoa_interf.append(calculate_angle_fft(aoa_map_interf))
            aoa_clean.append(calculate_angle_fft(aoa_map_clean))
            aoa_zero_mitigation.append(calculate_angle_fft(aoa_map_zero_mitigation))

            target_ranges = np.array([object_targets[p][4]]).flatten()
            target_velocities = np.array([object_targets[p][7]]).flatten()
            ta = np.array([object_targets[p][5]]).flatten()

            rd_o_masks, rd_n_masks, d_indices, v_indices = calculate_rd_object_and_noise_masks(target_ranges,
                                                                                               target_velocities,
                                                                                               num_fts,
                                                                                               dataset.num_ramps_per_packet)
            aoa_o_masks, aoa_n_masks = calculate_aoa_object_and_noise_masks(target_ranges, ta, num_fts,
                                                                            num_angle_fft_bins)
            target_angles.append({'d': d_indices, 'v': v_indices, 'a': ta})

            rd_object_masks.append(rd_o_masks)
            rd_noise_masks.append(rd_n_masks)

            aoa_object_masks.append(aoa_o_masks)
            aoa_noise_masks.append(aoa_n_masks)

    rd_object_masks = np.array(rd_object_masks)
    rd_noise_masks = np.array(rd_noise_masks)

    aoa_object_masks = np.array(aoa_object_masks)
    aoa_noise_masks = np.array(aoa_noise_masks)

    aoa_original = np.array(aoa_original)
    aoa_interf = np.array(aoa_interf)
    aoa_clean = np.array(aoa_clean)
    aoa_zero_mitigation = np.array(aoa_zero_mitigation)

    x = aoa_interf
    y = aoa_clean

    filter_mask = np.ones((x.shape[0],), dtype=int)
    interference_mask = filter_mask

    return x, y, aoa_clean, aoa_original, aoa_interf, filter_mask, interference_mask, aoa_zero_mitigation, rd_object_masks,\
           rd_noise_masks, aoa_object_masks, aoa_noise_masks, target_angles


class DataSource(Enum):
    DENOISE_REAL_IMAG_RAMP = load_data_for_denoising_ri_ramps
    DENOISE_REAL_IMAG_RD = load_data_for_denoising_ri_range_doppler_map
    DENOISE_REAL_IMAG_AOA = load_data_for_denoising_ri_angle_map
    DENOISE_LOG_MAG_RD = load_data_for_denoising_log_mag_range_doppler_map

    @staticmethod
    def from_name(value):
        if value == DataSource.DENOISE_REAL_IMAG_RAMP.__name__:
            return DataSource.DENOISE_REAL_IMAG_RAMP
        elif value == DataSource.DENOISE_REAL_IMAG_RD.__name__:
            return DataSource.DENOISE_REAL_IMAG_RD
        elif value == DataSource.DENOISE_REAL_IMAG_AOA.__name__:
            return DataSource.DENOISE_REAL_IMAG_AOA
        elif value == DataSource.DENOISE_LOG_MAG_RD.__name__:
            return DataSource.DENOISE_LOG_MAG_RD

    @staticmethod
    def data_content(value):
        if value is DataSource.DENOISE_REAL_IMAG_RAMP:
            return DataContent.COMPLEX_RAMP
        elif value is DataSource.DENOISE_REAL_IMAG_RD:
            return DataContent.COMPLEX_PACKET_RD
        elif value is DataSource.DENOISE_REAL_IMAG_AOA:
            return DataContent.COMPLEX_PACKET_AOA
        elif value is DataSource.DENOISE_LOG_MAG_RD:
            return DataContent.REAL_PACKET_RD


class DatasetPartition(Enum):
    TRAINING = 0
    VALIDATION = 1
    TEST = 2

    @staticmethod
    def mat_path_prefix(partition):
        if partition is DatasetPartition.TRAINING:
            return 'train'
        elif partition is DatasetPartition.VALIDATION:
            return 'val'
        elif partition is DatasetPartition.TEST:
            return 'test'


class DataContent(Enum):
    COMPLEX_RAMP = 1
    COMPLEX_PACKET_RD = 2
    COMPLEX_PACKET_AOA = 3
    REAL_PACKET_RD = 4

    @staticmethod
    def num_values_per_sample(data_content, num_fts, num_ramps_per_packet):
        if data_content is DataContent.COMPLEX_RAMP:
            return num_fts * 2
        elif data_content is DataContent.COMPLEX_PACKET_RD:
            return num_fts * num_ramps_per_packet * 2
        elif data_content is DataContent.COMPLEX_PACKET_AOA:
            return num_fts * 1024 * 2
        elif data_content is DataContent.REAL_PACKET_RD:
            return num_fts * num_ramps_per_packet

    @staticmethod
    def num_samples_per_packet(data_content, num_ramps_per_packet, num_antennas=8):
        if data_content is DataContent.COMPLEX_RAMP:
            return num_ramps_per_packet
        elif data_content is DataContent.COMPLEX_PACKET_RD:
            return 1
        elif data_content is DataContent.COMPLEX_PACKET_AOA:
            return 1/num_antennas
        elif data_content is DataContent.REAL_PACKET_RD:
            return 1

    @staticmethod
    def sample_shape(data_content, num_ramps_per_packet, num_fts):
        if data_content is DataContent.COMPLEX_RAMP:
            return 1, num_fts
        elif data_content is DataContent.COMPLEX_PACKET_RD:
            return num_fts, num_ramps_per_packet
        elif data_content is DataContent.COMPLEX_PACKET_AOA:
            return num_fts, 1024
        elif data_content is DataContent.REAL_PACKET_RD:
            return num_fts, num_ramps_per_packet

    @staticmethod
    def num_samples_for_rd_evaluation(data_content, num_ramps_per_packet):
        if data_content is DataContent.COMPLEX_RAMP:
            return num_ramps_per_packet
        elif data_content is DataContent.COMPLEX_PACKET_RD:
            return 1
        elif data_content is DataContent.REAL_PACKET_RD:
            return 1
        else:
            assert False

    @staticmethod
    def num_samples_for_aoa_evaluation(data_content, num_ramps_per_packet, num_channels):
        if data_content is DataContent.COMPLEX_RAMP:
            return num_ramps_per_packet * num_channels
        elif data_content is DataContent.COMPLEX_PACKET_AOA:
            return 1
        else:
            assert False


class RadarDataset(Dataset):

    """Radar dataset."""
    # # mat_path
    def __init__(self, data_source, mat_path, scaler, is_classification=False):
        """
        Args:
            mat_filename (string): Name of matlab mat file
        """

        if os.path.isdir('./data'):
            path_pref = './data'
        elif os.path.isdir('../data'):
            path_pref = '../data'
        else:
            assert False

        self.mat_path = os.path.join(path_pref, 'radar-data', mat_path)
        self.is_classification = is_classification
        self.data_content = DataSource.data_content(data_source)
        self.data_source = data_source

        mat_folder_path_train = os.path.join(self.mat_path, DatasetPartition.mat_path_prefix(DatasetPartition.TRAINING))
        mat_folder_path_val = os.path.join(self.mat_path, DatasetPartition.mat_path_prefix(DatasetPartition.VALIDATION))
        mat_folder_path_test = os.path.join(self.mat_path, DatasetPartition.mat_path_prefix(DatasetPartition.TEST))

        self.file_names = {DatasetPartition.TRAINING: os.listdir(mat_folder_path_train),
                           DatasetPartition.VALIDATION: os.listdir(mat_folder_path_val),
                           DatasetPartition.TEST: os.listdir(mat_folder_path_test)}

        self.file_names[DatasetPartition.TRAINING].sort()
        self.file_names[DatasetPartition.VALIDATION].sort()
        self.file_names[DatasetPartition.TEST].sort()

        self.sample_indices = {DatasetPartition.TRAINING: {},
                               DatasetPartition.VALIDATION: {},
                               DatasetPartition.TEST: {}}

        mean_s_IF_per_file = []
        var_s_IF_per_file = []
        mean_s_IF_clean_per_file = []
        var_s_IF_clean_per_file = []
        mean_s_IF_original_per_file = []
        var_s_IF_original_per_file = []

        cov_s_IF_per_file = []
        cov_s_IF_clean_per_file = []

        num_packets_in_train_files = []

        num_interfered_ramps = {DatasetPartition.TRAINING: 0,
                                DatasetPartition.VALIDATION: 0,
                                DatasetPartition.TEST: 0}

        num_interfered_samples = {DatasetPartition.TRAINING: 0,
                                  DatasetPartition.VALIDATION: 0,
                                  DatasetPartition.TEST: 0}

        self.num_samples = {DatasetPartition.TRAINING: 0,
                            DatasetPartition.VALIDATION: 0,
                            DatasetPartition.TEST: 0}

        self.num_channels = {DatasetPartition.TRAINING: 0,
                            DatasetPartition.VALIDATION: 0,
                            DatasetPartition.TEST: 0}

        print_()
        print_('# Reading data set meta data #')
        print_()
        print_('Data folder: {}'.format(mat_path))

        # read global config
        try:
            config_mat_path = os.path.join(self.mat_path, 'config.mat')
            print_('Reading config from {}'.format(config_mat_path))
            config = spio.loadmat(config_mat_path, squeeze_me=True)['config']

            self.num_channels_per_scene = config['radar'][()]['N_ant_rx'][()]
            self.num_ramps_per_packet = config['sig'][()]['N_sw'][()]
            num_td_samples_per_ramp = config['sig'][()]['N_samp_per_ramp'][()]
            self.num_fast_time_samples = int(num_td_samples_per_ramp / 2)
        except IOError:
            warnings.warn('IOError reading config.')
        except KeyError:
            warnings.warn(
                'KeyError reading config. File does not contain config struct. Skipping config, using default values...')
        except ValueError:
            warnings.warn('One or more config values missing.')

        self.num_samples_per_packet = DataContent.num_samples_per_packet(self.data_content, self.num_ramps_per_packet)

        for partition in self.file_names.keys():
            partition_path_prefix = DatasetPartition.mat_path_prefix(partition)

            # read partition-specific config
            try:
                config_mat_path = os.path.join(self.mat_path, partition_path_prefix + '-config.mat')
                print_('Reading partition-config from {}'.format(config_mat_path))
                config = spio.loadmat(config_mat_path, squeeze_me=True)['part_config']

                self.num_channels[partition] = config['num_ds_channels'][()]
            except IOError:
                warnings.warn('IOError reading config.')
            except KeyError:
                warnings.warn(
                    'KeyError reading config. File does not contain config struct. Skipping config, using default values...')
            except ValueError:
                warnings.warn('One or more config values missing.')

            for file_name in self.file_names[partition]:
                file_rel_path = os.path.join(partition_path_prefix, file_name)
                num_packets_in_file_str = file_name[file_name.find('_p') + 2: file_name.find('_c')]
                num_channels_in_file_str = file_name[file_name.find('_c') + 2: file_name.find('_i')]
                try:
                    num_packets_in_file = int(num_packets_in_file_str)
                    num_channels_in_file = int(num_channels_in_file_str)
                    assert(self.num_channels[partition] == num_channels_in_file)
                    num_samples_in_file = int(num_packets_in_file * num_channels_in_file * self.num_samples_per_packet)
                    self.sample_indices[partition][file_name] = (self.num_samples[partition], self.num_samples[partition] + num_samples_in_file - 1)
                    self.num_samples[partition] += num_samples_in_file
                except ValueError:
                    warnings.warn('Could not find number of packets contained in file {}'.format(file_rel_path))
                    print_('Skipping file {}. Num packets missing.'.format(file_name))
                    continue

                try:
                    print_('Loading {}'.format(file_rel_path))
                    out = spio.loadmat(os.path.join(self.mat_path, file_rel_path), squeeze_me=True)['out']
                except IOError:
                    warnings.warn('IOError reading file {}. Skipping file.'.format(file_rel_path))
                    continue
                except KeyError:
                    warnings.warn('KeyError reading file {}. File does not contain out struct. Skipping file.'.format(file_rel_path))
                    continue

                try:
                    num_interfered_ramps_in_file = out['num_interfered_ramps'][()]
                    num_interfered_ramps[partition] += num_interfered_ramps_in_file
                    num_interfered_samples_in_file = out['num_interfered_samples'][()]
                    num_interfered_samples[partition] += num_interfered_samples_in_file
                except ValueError:
                    warnings.warn('No info to num interfered ramps for file {}'.format(file_rel_path))

                if partition is DatasetPartition.TRAINING:
                    num_packets_in_train_files.append(num_packets_in_file)
                    try:
                        mean_s_IF_per_file.append(out['mean_s_IF'][()])
                        var_s_IF_per_file.append(out['var_s_IF'][()])
                        mean_s_IF_clean_per_file.append(out['mean_s_IF_clean'][()])
                        var_s_IF_clean_per_file.append(out['var_s_IF_clean'][()])
                        mean_s_IF_original_per_file.append(out['mean_s_IF_clean_noise'][()])
                        var_s_IF_original_per_file.append(out['var_s_IF_clean_noise'][()])
                    except ValueError:
                        warnings.warn('No mean / var data for file {}'.format(file_rel_path))

                    try:
                        cov_s_IF_per_file.append(out['cov_s_IF'][()])
                        cov_s_IF_clean_per_file.append(out['cov_s_IF_clean'][()])
                    except ValueError:
                        warnings.warn('No cov data for file {}'.format(file_rel_path))

            total_num_ramps = int(self.num_samples[partition] / self.num_samples_per_packet * self.num_ramps_per_packet)
            if total_num_ramps > 0:
                print_('Number interfered ramps for {}: {}/{} ({:.2f}%)'.format(partition,
                                                                               num_interfered_ramps[partition],
                                                                               total_num_ramps,
                                                                               100 / total_num_ramps * num_interfered_ramps[partition]))
            total_num_td_samples = self.num_fast_time_samples*2*self.num_ramps_per_packet*int(self.num_samples[partition] / self.num_samples_per_packet)
            if total_num_td_samples > 0:
                print_('Number interfered time domain samples for {}: {}/{} ({:.2f}%)'.format(partition,
                                                                                             num_interfered_samples[partition],
                                                                                             total_num_td_samples,
                                                                                             100 / total_num_td_samples * num_interfered_samples[partition]))

        print_()

        self.num_values_per_sample = DataContent.num_values_per_sample(self.data_content, self.num_fast_time_samples, self.num_ramps_per_packet)

        self.partition_indices = {DatasetPartition.TRAINING: list(range(0, self.num_samples[DatasetPartition.TRAINING])),
                                  DatasetPartition.VALIDATION: list(range(0, self.num_samples[DatasetPartition.VALIDATION])),
                                  DatasetPartition.TEST: list(range(0, self.num_samples[DatasetPartition.TEST]))}

        self.active_partition = DatasetPartition.TRAINING

        self.cached_samples = None
        self.cached_sample_indices = []

        self.scaler_x = None
        self.scaler_y = None

        if scaler is Scaler.STD_SCALER:  # Attention: clean / original depends on data source target!!!!
            self.fit_std_scaler(mean_s_IF_per_file, var_s_IF_per_file,
                                mean_s_IF_clean_per_file, var_s_IF_clean_per_file,  # mean_s_IF_original_per_file, var_s_IF_original_per_file,
                                num_packets_in_train_files, scaler)
        elif scaler is Scaler.COMPLEX_FEATURE_SCALER:
            self.fit_complex_feature_scaler(mean_s_IF_per_file, cov_s_IF_per_file,
                                            mean_s_IF_clean_per_file, cov_s_IF_clean_per_file,
                                            num_packets_in_train_files, scaler)

    def fit_std_scaler(self, mean_x_per_file, var_x_per_file,
                       mean_y_per_file, var_y_per_file,
                       num_packets_per_file, scaler):

        if not all(x == num_packets_per_file[0] for x in num_packets_per_file):
            warnings.warn('Not all files contain the same number of peckets. Scaling depends on this!!')
        assert (len(mean_x_per_file) == len(self.file_names[DatasetPartition.TRAINING]))

        self.scaler_x = scaler()

        num_files = len(mean_x_per_file)
        avg_num_packets_in_file = int(np.mean(num_packets_per_file))
        num_packets_for_scaler_fitting = np.sum(num_packets_per_file)

        mean_x = np.mean(mean_x_per_file)
        # calculate total variance from subset means and variances of same sample length
        # see https://stats.stackexchange.com/questions/10441/how-to-calculate-the-variance-of-a-partition-of-variables
        var_x = (avg_num_packets_in_file - 1) / (num_packets_for_scaler_fitting - 1) * \
                (np.sum(var_x_per_file + (
                            avg_num_packets_in_file * (num_files - 1) / (avg_num_packets_in_file - 1)) * np.var(
                    mean_x_per_file)))
        self.scaler_x.mean = mean_x
        self.scaler_x.var = var_x

        if not self.is_classification:
            self.scaler_y = scaler()

            mean_y = np.mean(mean_y_per_file)
            var_y = (avg_num_packets_in_file - 1) / (num_packets_for_scaler_fitting - 1) * \
                    (np.sum(var_y_per_file + (avg_num_packets_in_file * (num_files - 1) / (
                                avg_num_packets_in_file - 1)) * np.var(mean_y_per_file)))
            self.scaler_y.mean = mean_y
            self.scaler_y.var = var_y

    def fit_complex_feature_scaler(self, mean_x_per_file, cov_x_per_file,
                                   mean_y_per_file, cov_y_per_file,
                                   num_packets_per_file, scaler):

        if not all(x == num_packets_per_file[0] for x in num_packets_per_file):
            warnings.warn('Not all files contain the same number of peckets. Scaling depends on this!!')
        assert (len(mean_x_per_file) == len(self.file_names[DatasetPartition.TRAINING]))

        self.scaler_x = scaler()

        num_files = len(mean_x_per_file)
        avg_num_packets_in_file = int(np.mean(num_packets_per_file))
        num_packets_for_scaler_fitting = np.sum(num_packets_per_file)

        mean_x = np.mean(mean_x_per_file)
        # calculate total variance from subset means and variances of same sample length
        # see https://stats.stackexchange.com/questions/10441/how-to-calculate-the-variance-of-a-partition-of-variables
        cov_x = (avg_num_packets_in_file - 1) / (num_packets_for_scaler_fitting - 1) * \
                   (np.sum(cov_x_per_file + (avg_num_packets_in_file * (num_files - 1) / (avg_num_packets_in_file - 1)) * np.var(mean_x_per_file), axis=0))
        sr_cov = scipy.linalg.sqrtm(cov_x)
        inv_sr_cov = np.linalg.inv(sr_cov)

        self.scaler_x.mean_complex = mean_x
        self.scaler_x.sr_cov = sr_cov
        self.scaler_x.inv_sr_cov = inv_sr_cov

        if not self.is_classification:
            self.scaler_y = scaler()

            mean_y = np.mean(mean_y_per_file)
            cov_y = (avg_num_packets_in_file - 1) / (num_packets_for_scaler_fitting - 1) * \
                    (np.sum(cov_y_per_file + (avg_num_packets_in_file * (num_files - 1) / (
                                avg_num_packets_in_file - 1)) * np.var(mean_y_per_file), axis=0))
            sr_cov = scipy.linalg.sqrtm(cov_y)
            inv_sr_cov = np.linalg.inv(sr_cov)
            self.scaler_y.mean_complex = mean_y
            self.scaler_y.sr_cov = sr_cov
            self.scaler_y.inv_sr_cov = inv_sr_cov

    def get_sample_start_and_end_indices_per_file(self):
        if len(self.partition_indices[self.active_partition]) == 0:
            return []
        sample_start_end_indices = []
        for file_name in self.file_names[self.active_partition]:
            sample_start_end_indices.append(self.sample_indices[self.active_partition][file_name])

        return sample_start_end_indices

    def get_num_channels(self):
        return self.num_channels[self.active_partition]

    def __len__(self):
        return len(self.partition_indices[self.active_partition])

    def __getitem__(self, idx):
        x, y, _, _, _, filter_mask, _, _, rd_object_masks, rd_noise_masks, aoa_object_masks, aoa_noise_masks, _ = self.load_data_for_sample_from_cache_or_disk(idx)

        x = self.scale(x, is_y=False)[0]
        y = self.scale(y, is_y=True)[0]

        x = complex_to_format(self.data_content, x)
        y = complex_to_format(self.data_content, y)

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        filter_mask = torch.tensor(filter_mask.flatten()[0], dtype=torch.uint8)

        object_mask = torch.zeros(y.size())
        if self.data_source in [DataSource.DENOISE_REAL_IMAG_RD, DataSource.DENOISE_LOG_MAG_RD]:
            object_mask = torch.tensor(rd_object_masks[0])
        elif self.data_source is DataSource.DENOISE_REAL_IMAG_AOA:
            object_mask = torch.tensor(aoa_object_masks[0])

        noise_mask = torch.ones(y.size())
        if self.data_source in [DataSource.DENOISE_REAL_IMAG_RD, DataSource.DENOISE_LOG_MAG_RD]:
            noise_mask = torch.tensor(rd_noise_masks[0])
        elif self.data_source is DataSource.DENOISE_REAL_IMAG_AOA:
            noise_mask = torch.tensor(aoa_noise_masks[0])

        return x, y, filter_mask, object_mask, noise_mask

    def load_data_for_sample_from_cache_or_disk(self, sample_idx):

        try:
            cached_sample_idx = self.cached_sample_indices.index(sample_idx)
            return self.sample_at_index_from_cache(cached_sample_idx)
        except ValueError:
            pass
        # item not cached --> load from file

        for fn in self.file_names[self.active_partition]:
            (start_i, end_i) = self.sample_indices[self.active_partition][fn]
            if start_i <= sample_idx <= end_i:
                file_name = fn
                start_idx = start_i
                end_idx = end_i
                break

        out = spio.loadmat(os.path.join(self.mat_path, DatasetPartition.mat_path_prefix(self.active_partition), file_name), squeeze_me=True)['out']
        self.cached_samples = self.data_source(out, self)
        self.cached_sample_indices = list(range(start_idx, end_idx + 1))

        cached_sample_idx = self.cached_sample_indices.index(sample_idx)
        return self.sample_at_index_from_cache(cached_sample_idx)

    def sample_at_index_from_cache(self, sample_idx):
        sample_shape = DataContent.sample_shape(self.data_content, self.num_ramps_per_packet, self.num_fast_time_samples)
        sample_shape_batch = (1, sample_shape[0], sample_shape[1])
        target_idx = min(sample_idx, int(sample_idx / self.num_samples_per_packet))
        return (self.cached_samples[0][sample_idx:sample_idx+1].reshape(sample_shape_batch),
                self.cached_samples[1][sample_idx:sample_idx+1].reshape(sample_shape_batch),
                self.cached_samples[2][sample_idx:sample_idx+1].reshape(sample_shape_batch),
                self.cached_samples[3][sample_idx:sample_idx+1].reshape(sample_shape_batch),
                self.cached_samples[4][sample_idx:sample_idx+1].reshape(sample_shape_batch),
                self.cached_samples[5][sample_idx:sample_idx+1].reshape(1),
                self.cached_samples[6][sample_idx:sample_idx+1].reshape(1),
                self.cached_samples[7][sample_idx:sample_idx+1].reshape(sample_shape_batch),
                self.cached_samples[8][target_idx:target_idx + 1].reshape((1, self.num_fast_time_samples, self.num_ramps_per_packet)),
                self.cached_samples[9][target_idx:target_idx+1].reshape((1, self.num_fast_time_samples, self.num_ramps_per_packet)),
                self.cached_samples[10][target_idx:target_idx + 1].reshape((1, self.num_fast_time_samples, num_angle_fft_bins)),
                self.cached_samples[11][target_idx:target_idx + 1].reshape((1, self.num_fast_time_samples, num_angle_fft_bins)),
                self.cached_samples[12][target_idx:target_idx + 1])

    def get_scene_rd_object_and_noise_masks(self, scene_idx):
        num_samples = DataContent.num_samples_for_rd_evaluation(self.data_content, self.num_ramps_per_packet)
        scene_start_idx = scene_idx * num_samples
        _, _, _, _, _, _, _, _, rd_object_masks, rd_noise_masks, _, _, _ = self.load_data_for_sample_from_cache_or_disk(scene_start_idx)

        return rd_object_masks[0].astype(bool), rd_noise_masks[0].astype(bool)

    def get_scene_rd_clean(self, scene_idx):
        num_samples = DataContent.num_samples_for_rd_evaluation(self.data_content, self.num_ramps_per_packet)
        scene_start_idx = scene_idx * num_samples
        scene_end_idx = (scene_idx + 1) * num_samples
        _, _, packet_data, _, _, _, _, _, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(scene_start_idx)
        for i in range(scene_start_idx+1, scene_end_idx):
            _, _, packet_data_i, _, _, _, _, _, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(i)
            packet_data = np.vstack((packet_data, packet_data_i))
        if self.data_source == DataSource.DENOISE_REAL_IMAG_RAMP:
            packet_data = calculate_velocity_fft(packet_data[:, 0, :])
        else:
            packet_data = packet_data[0]
        return packet_data

    def get_scene_rd_original(self, scene_idx):
        num_samples = DataContent.num_samples_for_rd_evaluation(self.data_content, self.num_ramps_per_packet)
        scene_start_idx = scene_idx * num_samples
        scene_end_idx = (scene_idx + 1) * num_samples
        _, _, _, packet_data, _, _, _, _, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(scene_start_idx)
        for i in range(scene_start_idx + 1, scene_end_idx):
            _, _, _, packet_data_i, _, _, _, _, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(i)
            packet_data = np.vstack((packet_data, packet_data_i))
        if self.data_source == DataSource.DENOISE_REAL_IMAG_RAMP:
            packet_data = calculate_velocity_fft(packet_data[:, 0, :])
        else:
            packet_data = packet_data[0]
        return packet_data

    def get_scene_rd_interf(self, scene_idx):
        num_samples = DataContent.num_samples_for_rd_evaluation(self.data_content, self.num_ramps_per_packet)
        scene_start_idx = scene_idx * num_samples
        scene_end_idx = (scene_idx + 1) * num_samples
        _, _, _, _, packet_data, _, _, _, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(scene_start_idx)
        for i in range(scene_start_idx + 1, scene_end_idx):
            _, _, _, _, packet_data_i, _, _, _, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(i)
            packet_data = np.vstack((packet_data, packet_data_i))
        if self.data_source == DataSource.DENOISE_REAL_IMAG_RAMP:
            packet_data = calculate_velocity_fft(packet_data[:, 0, :])
        else:
            packet_data = packet_data[0]
        return packet_data

    def get_scene_rd_zero_substitude_in_time_domain(self, scene_idx):
        num_samples = DataContent.num_samples_for_rd_evaluation(self.data_content, self.num_ramps_per_packet)
        scene_start_idx = scene_idx * num_samples
        scene_end_idx = (scene_idx + 1) * num_samples
        _, _, _, _, _, _, _, packet_data, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(scene_start_idx)
        for i in range(scene_start_idx + 1, scene_end_idx):
            _, _, _, _, _, _, _, packet_data_i, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(i)
            packet_data = np.vstack((packet_data, packet_data_i))
        if self.data_source == DataSource.DENOISE_REAL_IMAG_RAMP:
            packet_data = calculate_velocity_fft(packet_data[:, 0, :])
        else:
            packet_data = packet_data[0]
        return packet_data

    def get_scene_cr_object_and_noise_masks(self, scene_idx, d_idx, v_idx):
        num_channels = self.get_num_channels()
        num_samples = DataContent.num_samples_for_rd_evaluation(self.data_content, self.num_ramps_per_packet)
        scene_start_idx = scene_idx * num_samples
        assert (num_channels > 1)
        _, _, _, _, _, _, _, _, rd_object_mask, rd_noise_mask, aoa_object_masks, aoa_noise_masks, target_angles = self.load_data_for_sample_from_cache_or_disk(
            scene_start_idx)

        cr_object_masks, cr_noise_masks = calculate_cr_object_and_noise_masks(target_angles[0], d_idx, v_idx, num_angle_fft_bins)

        return cr_object_masks.astype(bool), cr_noise_masks.astype(bool)

    def get_scene_aoa_object_and_noise_masks(self, scene_idx):
        num_channels = self.get_num_channels()
        num_samples = DataContent.num_samples_for_aoa_evaluation(self.data_content, self.num_ramps_per_packet, num_channels)
        scene_start_idx = scene_idx * num_samples
        assert (num_channels > 1)
        _, _, _, _, _, _, _, _, _, _, aoa_object_masks, aoa_noise_masks, _ = self.load_data_for_sample_from_cache_or_disk(
            scene_start_idx)

        return aoa_object_masks[0].astype(bool), aoa_noise_masks[0].astype(bool)

    def get_scene_aoa_clean(self, scene_idx):
        if self.data_source == DataSource.DENOISE_REAL_IMAG_RAMP:
            assert(self.num_channels[self.active_partition] > 1)
            num_samples = DataContent.num_samples_for_aoa_evaluation(self.data_content, self.num_ramps_per_packet, self.num_channels[self.active_partition])
            scene_start_idx = scene_idx * num_samples
            scene_end_idx = (scene_idx + 1) * num_samples
            packet_data = np.zeros((self.num_fast_time_samples, self.num_channels[self.active_partition]), dtype=np.complex128)
            channel = 0
            for i in range(scene_start_idx, scene_end_idx, self.num_ramps_per_packet):
                _, _, packet_data_i, _, _, _, _, _, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(i)
                packet_data[:, channel] = packet_data_i.reshape(packet_data.shape[0])
                channel += 1
            packet_data = calculate_angle_fft(packet_data)
        else:
            _, _, packet_data, _, _, _, _, _, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(scene_idx)
            packet_data = packet_data[0]
        return packet_data

    def get_scene_aoa_original(self, scene_idx):
        if self.data_source == DataSource.DENOISE_REAL_IMAG_RAMP:
            assert (self.num_channels[self.active_partition] > 1)
            num_samples = DataContent.num_samples_for_aoa_evaluation(self.data_content, self.num_ramps_per_packet, self.num_channels[self.active_partition])
            scene_start_idx = scene_idx * num_samples
            scene_end_idx = (scene_idx + 1) * num_samples
            packet_data = np.zeros((self.num_fast_time_samples, self.num_channels[self.active_partition]), dtype=np.complex128)
            channel = 0
            for i in range(scene_start_idx, scene_end_idx, self.num_ramps_per_packet):
                _, _, _, packet_data_i, _, _, _, _, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(i)
                packet_data[:, channel] = packet_data_i.reshape(packet_data.shape[0])
                channel += 1
            packet_data = calculate_angle_fft(packet_data)
        else:
            _, _, _, packet_data, _, _, _, _, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(scene_idx)
            packet_data = packet_data[0]
        return packet_data

    def get_scene_aoa_interf(self, scene_idx):
        if self.data_source == DataSource.DENOISE_REAL_IMAG_RAMP:
            assert (self.num_channels[self.active_partition] > 1)
            num_samples = DataContent.num_samples_for_aoa_evaluation(self.data_content, self.num_ramps_per_packet, self.num_channels[self.active_partition])
            scene_start_idx = scene_idx * num_samples
            scene_end_idx = (scene_idx + 1) * num_samples
            packet_data = np.zeros((self.num_fast_time_samples, self.num_channels[self.active_partition]), dtype=np.complex128)
            channel = 0
            for i in range(scene_start_idx, scene_end_idx, self.num_ramps_per_packet):
                _, _, _, _, packet_data_i, _, _, _, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(i)
                packet_data[:, channel] = packet_data_i.reshape(packet_data.shape[0])
                channel += 1
            packet_data = calculate_angle_fft(packet_data)
        else:
            _, _, _, _, packet_data, _, _, _, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(scene_idx)
            packet_data = packet_data[0]
        return packet_data

    def get_scene_aoa_zero_substitude_in_time_domain(self, scene_idx):
        if self.data_source == DataSource.DENOISE_REAL_IMAG_RAMP:
            assert (self.num_channels[self.active_partition] > 1)
            num_samples = DataContent.num_samples_for_aoa_evaluation(self.data_content, self.num_ramps_per_packet, self.num_channels[self.active_partition])
            scene_start_idx = scene_idx * num_samples
            scene_end_idx = (scene_idx + 1) * num_samples
            packet_data = np.zeros((self.num_fast_time_samples, self.num_channels[self.active_partition]), dtype=np.complex128)
            channel = 0
            for i in range(scene_start_idx, scene_end_idx, self.num_ramps_per_packet):
                _, _, _, _, _, _, _, packet_data_i, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(i)
                packet_data[:, channel] = packet_data_i.reshape(packet_data.shape[0])
                channel += 1
            packet_data = calculate_angle_fft(packet_data)
        else:
            _, _, _, _, _, _, _, packet_data, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(scene_idx)
            packet_data = packet_data[0]
        return packet_data

    def get_sample_interference_mask(self, scene_idx, num_samples_per_scene):
        scene_start_idx = scene_idx * num_samples_per_scene
        scene_end_idx = (scene_idx + 1) * num_samples_per_scene
        _, _, _, _, _, _, packet_data, _, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(scene_start_idx)
        for i in range(scene_start_idx+1, scene_end_idx):
            _, _, _, _, _, _, packet_data_i, _, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(i)
            packet_data = np.vstack((packet_data, packet_data_i))
        return packet_data

    def get_target_original_scaled_re_im(self, scene_idx, num_samples_per_scene):
        scene_start_idx = scene_idx * num_samples_per_scene
        scene_end_idx = (scene_idx + 1) * num_samples_per_scene
        _, _, _, packet_data, _, _, _, _, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(scene_start_idx)
        for i in range(scene_start_idx+1, scene_end_idx):
            _, _, _, packet_data_i, _, _, _, _, _, _, _, _, _ = self.load_data_for_sample_from_cache_or_disk(i)
            packet_data = np.vstack((packet_data, packet_data_i))
        packet_data = self.scale(packet_data, is_y=True)
        packet_data = self.packet_complex_to_target_format(packet_data)
        return packet_data

    def clone_for_new_active_partition(self, partition: DatasetPartition):
        clone = copy.deepcopy(self)
        clone.active_partition = partition
        clone.cached_samples = None
        clone.cached_sample_indices = []
        return clone

    def inverse_scale(self, data, is_y):
        if is_y and self.is_classification:
            return data

        if is_y:
            scaler = self.scaler_y
        else:
            scaler = self.scaler_x

        if scaler is not None:
            data = scaler.inverse_transform(data)
        return data

    def scale(self, data, is_y):
        if is_y and self.is_classification:
            return data

        if is_y:
            scaler = self.scaler_y
        else:
            scaler = self.scaler_x

        if scaler is not None:
            data = scaler.transform(data)
        return data

    def packet_in_target_format_to_complex(self, packet_data, packet_idx=None):
        if self.data_content is DataContent.COMPLEX_RAMP:
            packet_data = packet_data[:, :, :self.num_fast_time_samples] + 1j * packet_data[:, :, self.num_fast_time_samples:]
        elif self.data_content is DataContent.COMPLEX_PACKET_RD:
            packet_data = packet_data[:, :, :self.num_ramps_per_packet] + 1j * packet_data[:, :, self.num_ramps_per_packet:]
        elif self.data_content is DataContent.COMPLEX_PACKET_AOA:
            packet_data = packet_data[:, :, :1024] + 1j * packet_data[:, :, 1024:]
        elif self.data_content is DataContent.REAL_PACKET_RD:
            pass
        else:
            assert False
        return packet_data

    def packet_complex_to_target_format(self, packet_data):
        return complex_to_format(self.data_content, packet_data)


def complex_to_format(target_content, data):
    if target_content is DataContent.COMPLEX_RAMP:
        last_axis = len(data.shape)-1
        data = np.concatenate((np.real(data), np.imag(data)), axis=last_axis)
    elif target_content is DataContent.COMPLEX_PACKET_RD:
        last_axis = len(data.shape) - 1
        data = np.concatenate((np.real(data), np.imag(data)), axis=last_axis)
    elif target_content is DataContent.COMPLEX_PACKET_AOA:
        last_axis = len(data.shape) - 1
        data = np.concatenate((np.real(data), np.imag(data)), axis=last_axis)
    elif target_content is DataContent.REAL_PACKET_RD:
        pass
    else:
        assert False
    return data


def calculate_rd_object_and_noise_masks(target_distances, target_velocities, num_fts, num_ramps):
    d_vec = d_vec_fft2(num_fts)
    v_vec = v_vec_fft2(num_ramps)
    return calculate_object_and_noise_masks(target_distances, target_velocities, num_fts, num_ramps, v_vec, d_vec)


def calculate_aoa_object_and_noise_masks(target_distances, target_angles, num_fts, num_fft3_bins):
    a_vec = np.arcsin(1 * np.linspace(-1, 1, num_angle_fft_bins))
    d_vec = np.linspace(0, 1, num_angle_fft_bins) * d_max
    obj_masks, noise_masks, _, _ = calculate_object_and_noise_masks(target_distances, target_angles, num_fts, num_fft3_bins, a_vec, d_vec)
    return obj_masks, noise_masks


def calculate_object_and_noise_masks(target_rows, target_columns, shape0, shape1, x_vec, y_vec, noise_radius=3):
    obj_mask = np.zeros((shape0, shape1), dtype=np.uint8)
    noise_mask = np.ones((shape0, shape1), dtype=np.uint8)

    target_range_indices = []
    for r in target_rows:
        target_range_indices.append(np.argmin(np.abs(y_vec - r)))

    target_velocity_indices = []
    for v in target_columns:
        target_velocity_indices.append(np.argmin(np.abs(x_vec - v)))

    obj_mask[target_range_indices, target_velocity_indices] = 1

    for i in range(len(target_range_indices)):
        r = target_range_indices[i]
        v = target_velocity_indices[i]
        r_min = max(r - noise_radius, 0)
        r_max = min(r + noise_radius + 1, shape0)
        v_min = max(v - noise_radius, 0)
        v_max = min(v + noise_radius + 1, shape1)

        noise_mask[r_min:r_max, v_min:v_max] = False

    return obj_mask, noise_mask, target_range_indices, target_velocity_indices


def calculate_cr_object_and_noise_masks(target_angles, d_idx, v_idx, num_angle_bins, noise_radius=70):
    obj_mask = np.zeros((1, num_angle_bins), dtype=np.uint8)
    noise_mask = np.ones((1, num_angle_bins), dtype=np.uint8)

    a_vec = np.arcsin(1 * np.linspace(-1, 1, num_angle_fft_bins))

    target_cross_range_index = None
    d_indices = target_angles['d']
    v_indices = target_angles['v']
    angles = target_angles['a']

    for i in range(len(angles)):
        if d_indices[i] == d_idx and v_indices[i] == v_idx:
            target_cross_range_index = np.argmin(np.abs(a_vec - angles[i]))
            break

    if target_cross_range_index is None:
        assert False

    obj_mask[0, target_cross_range_index] = 1

    cr_min = max(target_cross_range_index - noise_radius, 0)
    cr_max = min(target_cross_range_index + noise_radius + 1, num_angle_bins)
    noise_mask[0, cr_min:cr_max] = False

    return obj_mask, noise_mask
