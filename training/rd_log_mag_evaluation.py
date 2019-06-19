import time

import numpy as np
import torch
from datasets.radar_dataset import DataSource, DataContent
from training.evaluation_commons import Signal, EvaluationFunction
from utils.rd_processing import calculate_velocity_fft
from run_scripts import device
from run_scripts import verbose, visualize, RESIDUAL_LEARNING, print_
from utils.loading import data_loader_for_dataset
from utils.plotting import plot_rd_matrix_for_packet, plot_object_mag_cuts, \
    plot_values, plot_rd_noise_mask
from utils.printing import print_evaluation_summary


def evaluate_rd_log_mag(model, dataset, phase):
    print_('# # Evaluation: {} # #'.format(phase))

    num_channels = dataset.get_num_channels()

    # dataloader with batch_size = number of samples per packet --> evaluate one packet at once for evaluation metrics
    num_samples_for_evaluation_metrics = DataContent.num_samples_for_rd_evaluation(dataset.data_content, dataset.num_ramps_per_packet)
    dataloader = data_loader_for_dataset(dataset, num_samples_for_evaluation_metrics, shuffle=False)

    since_test = time.time()
    model.eval()
    packet_idx = 0

    signals_for_quality_measures = {
        Signal.PREDICTION: None,
        Signal.PREDICTION_SUBSTITUDE: None,
        Signal.CLEAN: None,
        Signal.INTERFERED: None,
        Signal.CLEAN_NOISE: None,
        Signal.BASELINE_ZERO_SUB: None
    }

    functions_for_quality_measures = [EvaluationFunction.SINR_RD_LOG_MAG, EvaluationFunction.PEAK_MAG_MSE]
    quality_measures = np.zeros((len(functions_for_quality_measures), len(signals_for_quality_measures), len(dataloader)))

    for inputs, labels, filter_mask, _, _ in dataloader:

        inputs = inputs[filter_mask].to(device)
        interference_mask = torch.tensor(dataset.get_sample_interference_mask(packet_idx, num_samples_for_evaluation_metrics), dtype=torch.uint8)[filter_mask]

        if inputs.shape[0] == 0:
            packet_idx += 1
            continue

        prediction_interf_substi = np.array(dataset.get_target_original_scaled_re_im(packet_idx, num_samples_for_evaluation_metrics), copy=True)
        prediction = np.zeros(prediction_interf_substi.shape)

        original_indices = np.nonzero(filter_mask.numpy())[0]

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

        if RESIDUAL_LEARNING:
            packet_prediction = inputs.cpu().numpy() - outputs.cpu().numpy()
        else:
            packet_prediction = outputs.cpu().numpy()

        for i in range(len(outputs)):
            original_index = original_indices[i]
            prediction[original_index] = packet_prediction[i]
            if interference_mask[i]:
                prediction_interf_substi[original_index] = packet_prediction[i]

        prediction_ri = dataset.packet_in_target_format_to_complex(prediction, packet_idx)
        prediction_original_scale_ri = dataset.inverse_scale(prediction_ri, is_y=True)

        prediction_interf_substi_ri = dataset.packet_in_target_format_to_complex(prediction_interf_substi, packet_idx)
        prediction_interf_substi_original_scale_ri = dataset.inverse_scale(prediction_interf_substi_ri, is_y=True)

        if dataset.data_source == DataSource.DENOISE_REAL_IMAG_RAMP:
            prediction_rd = calculate_velocity_fft(prediction_original_scale_ri[:, 0, :])
            prediction_substi_rd = calculate_velocity_fft(prediction_interf_substi_original_scale_ri[:, 0, :])
        else:
            prediction_rd = prediction_original_scale_ri[0]
            prediction_substi_rd = prediction_interf_substi_original_scale_ri[0]

        clean_rd = dataset.get_scene_rd_clean(packet_idx)
        interf_rd = dataset.get_scene_rd_interf(packet_idx)
        clean_noise_rd = dataset.get_scene_rd_original(packet_idx)
        zero_substi_baseline_rd = dataset.get_scene_rd_zero_substitude_in_time_domain(packet_idx)
        target_object_mask, target_noise_mask = dataset.get_scene_rd_object_and_noise_masks(packet_idx)

        signals_for_quality_measures[Signal.PREDICTION] = prediction_rd
        signals_for_quality_measures[Signal.PREDICTION_SUBSTITUDE] = prediction_substi_rd
        signals_for_quality_measures[Signal.CLEAN] = clean_rd
        signals_for_quality_measures[Signal.INTERFERED] = interf_rd
        signals_for_quality_measures[Signal.CLEAN_NOISE] = clean_noise_rd
        signals_for_quality_measures[Signal.BASELINE_ZERO_SUB] = zero_substi_baseline_rd

        for i, func in enumerate(functions_for_quality_measures):
            for j, signal_name in enumerate(signals_for_quality_measures):
                signal = signals_for_quality_measures[signal_name]
                quality_measures[i, j, packet_idx] = func(clean_rd, signal, target_object_mask, target_noise_mask)

        scene_idx = int(packet_idx / num_channels)

        if visualize and scene_idx in [0, 1] and packet_idx % num_channels == 0:
            plot_rd_matrix_for_packet(
                clean_rd,
                prediction_rd,
                prediction_substi_rd,
                interf_rd,
                zero_substi_baseline_rd,
                phase, packet_idx,
                dataset.data_source == DataSource.DENOISE_REAL_IMAG_RAMP,
                is_log_mag=True)

            object_mask, noise_mask = dataset.get_scene_rd_object_and_noise_masks(packet_idx)

            plot_rd_noise_mask(noise_mask, 'RD Noise mask', 'eval_{}_rd_noise_mask_p{}'.format(phase, packet_idx))

            plot_object_mag_cuts(prediction_rd,
                                 prediction_substi_rd,
                                 clean_rd,
                                 clean_noise_rd,
                                 interf_rd,
                                 zero_substi_baseline_rd,
                                 object_mask,
                                 packet_idx,
                                 phase,
                                 is_rd=True,
                                 is_log_mag=True)

        packet_idx += 1

    time_test = time.time() - since_test

    metrics = []
    metric_labels = []
    main_metric = None

    for i, func in enumerate(functions_for_quality_measures):
        func_metrics = []
        func_metric_labels = []
        sig_labels = []
        for j, signal in enumerate(signals_for_quality_measures):
            quality_measures_per_func_sign = quality_measures[i, j]
            count_nans = np.count_nonzero(np.isnan(quality_measures_per_func_sign))
            if count_nans > 0:
                quality_measures_per_func_sign = quality_measures_per_func_sign[np.logical_not(np.isnan(quality_measures_per_func_sign))]
                print_('WARNING: quality measure "{}" produces {} nans!!'.format(func.label(), count_nans))
            func_metrics.append(np.mean(quality_measures_per_func_sign))
            metric_label = '{} {}'.format(func.label(), signal.label())
            func_metric_labels.append(metric_label)
            sig_labels.append(signal.label())

            if func is EvaluationFunction.SINR_RD_LOG_MAG and signal is Signal.PREDICTION:
                main_metric = func_metrics[-1]

        plot_values(quality_measures[i], sig_labels, func.label(), phase)
        metrics.extend(func_metrics)
        metric_labels.extend(func_metric_labels)

    if verbose:
        print_evaluation_summary(time_test, phase, metrics, metric_labels)

    return main_metric
