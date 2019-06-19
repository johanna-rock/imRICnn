from utils.rd_processing import v_vec_fft2, basis_vec_fft3, d_max, calculate_cross_range_fft
from run_scripts import visualize, task_id, JOB_DIR
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
import numpy as np

ZOOM_LIMIT = 2048
FIG_SIZE = (16, 8)
FIG_SIZE_HIGH = (8, 16)
FIG_SIZE_SINGLE = (8, 8)
STD_CMAP = 'winter'  # 'nipy_spectral'
color_scale_max = 0
color_scale_min = -70

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

plt.style.use("ggplot")


def save_or_show_plot(name1, name2='', force_show=False, export_tikz=False):
    if not visualize and not force_show:
        return
    filename = JOB_DIR + '/' + name1 + '_id' + str(task_id) + name2
    plt.savefig(filename + '.png')
    if export_tikz:
        tikz_save(filename + '.tex')
    plt.close()


def plot_target_and_prediction(targets, epoch, num_epochs, phase, predictions):
    if not visualize:
        return
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)
    plt.plot(targets, label='target')
    plt.plot(predictions, label='prediction')
    plt.legend()
    plt.title("{} targets and prediction; epoch {}/{}".format(phase, epoch, num_epochs))
    ax.ticklabel_format(useOffset=False, style='plain')
    save_or_show_plot('visual_target+predict', '_' + phase + '_epoch' + str(epoch))

    if len(targets) > ZOOM_LIMIT:
        fig = plt.figure(figsize=FIG_SIZE)
        ax = fig.add_subplot(111)
        plt.plot(targets[ZOOM_LIMIT:2*ZOOM_LIMIT], label='target')
        plt.plot(predictions[ZOOM_LIMIT:2*ZOOM_LIMIT], label='prediction')
        plt.legend()
        plt.title("Zoom: {} targets and prediction; epoch {}/{}".format(phase, epoch, num_epochs))
        ax.ticklabel_format(useOffset=False, style='plain')
        save_or_show_plot('visual_target+predict', '_' + phase + '_epoch' + str(epoch) + '_zoom')


def plot_losses(losses):
    if not visualize:
        return
    plt.figure(figsize=FIG_SIZE)
    for phase in ['train', 'val']:
        plt.plot(losses[phase], label='phase:' + phase)
    plt.legend()
    plt.title("train and val losses")
    save_or_show_plot('visual_losses')


def plot_input_data(dataloaders, dataset_sizes):
    if not visualize:
        return
    # plot first element of all input windows
    plt.figure(figsize=FIG_SIZE)
    x = np.arange(0, dataset_sizes['train'])
    y = dataloaders['train'].dataset.x.numpy()[:, 0]
    plt.plot(x, y, label='training')
    if dataset_sizes['val'] > 0:
        x = np.arange(dataset_sizes['train'], dataset_sizes['train'] + dataset_sizes['val'])
        y = dataloaders['val'].dataset.x.numpy()[:, 0]
        plt.plot(x, y, label='val')
    plt.legend()
    plt.title("First element of input window per sample")
    save_or_show_plot('visual_inputs_first')
    # plot first n windows
    plt.figure(figsize=FIG_SIZE)
    n = 10
    for i in range(n):
        values = dataloaders['train'].dataset.x.numpy()[i, :]
        if len(values) == 1:
            plt.scatter(np.arange(2 * i * len(values), (2 * i + 1) * len(values)), values, s=1.5, marker='o', label='window ' + str(i))
        else:
            plt.plot(np.arange(2 * i * len(values), (2 * i + 1) * len(values)), values, label='window ' + str(i))
    plt.legend()
    plt.title("Total input for first {} samples".format(n))
    save_or_show_plot('visual_input_windows')


def plot_data_targets_predictions(phase, data, targets, predictions, title_add, filename_add=''):
    if not visualize:
        return
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(311)
    fig.suptitle("Evaluation ({} - {})".format(phase, title_add))
    plt.plot(np.real(data.reshape(-1)), label='data real')
    plt.plot(np.imag(data.reshape(-1)), label='data imag')
    plt.legend()
    ax.set_title("Data")
    ax.ticklabel_format(useOffset=False, style='plain')
    ax = fig.add_subplot(312)
    plt.plot(np.real(targets.reshape(-1)), label='targets')
    plt.plot(np.real(predictions.reshape(-1)), label='predictions')
    plt.legend()
    ax.set_title("Targets & Prediction real part")
    ax.ticklabel_format(useOffset=False, style='plain')
    ax = fig.add_subplot(313)
    plt.plot(np.imag(targets.reshape(-1)), label='targets')
    plt.plot(np.imag(predictions.reshape(-1)), label='predictions')
    plt.legend()
    ax.set_title("Targets & Prediction imag part")
    ax.ticklabel_format(useOffset=False, style='plain')
    save_or_show_plot('eval_{}_sig+target+predict_{}'.format(phase, filename_add))


def plot_interfered_original_clean_data(interfered_data, original_data, clean_data, packet):
    if not visualize or task_id > 0:
        return
    fig = plt.figure(figsize=FIG_SIZE)
    fig.suptitle("Data fft1 (p{})".format(packet))

    ax = fig.add_subplot(311)
    plt.plot(np.real(interfered_data.reshape(-1)), label='data real')
    plt.plot(np.imag(interfered_data.reshape(-1)), label='data imag')
    plt.legend()
    ax.set_title("Interfered Data")
    ax.ticklabel_format(useOffset=False, style='plain')

    ax = fig.add_subplot(312)
    plt.plot(np.real(original_data.reshape(-1)), label='data real')
    plt.plot(np.imag(original_data.reshape(-1)), label='data imag')
    plt.legend()
    ax.set_title("Original Data")
    ax.ticklabel_format(useOffset=False, style='plain')

    ax = fig.add_subplot(313)
    plt.plot(np.real(clean_data.reshape(-1)), label='data real')
    plt.plot(np.imag(clean_data.reshape(-1)), label='data imag')
    plt.legend()
    ax.set_title("Clean Data")
    ax.ticklabel_format(useOffset=False, style='plain')

    save_or_show_plot('data_int_orig_clean_p{}'.format(packet))


def plot_data(phase, data):
    if not visualize:
        return
    fig = plt.figure(figsize=FIG_SIZE_SINGLE)
    ax = fig.add_subplot(111)
    fig.suptitle("Evaluation ({})".format(phase))
    plt.plot(np.real(data.reshape(-1)), label='data real')
    plt.plot(np.imag(data.reshape(-1)), label='data imag')
    plt.legend()
    ax.set_title("Data")
    ax.ticklabel_format(useOffset=False, style='plain')
    save_or_show_plot('data_{}_sig'.format(phase))


def plot_classification_targets_and_predictions(phase, targets, predictions):
    if not visualize:
        return
    fig = plt.figure(figsize=FIG_SIZE_SINGLE)
    ax = fig.add_subplot(211)
    fig.suptitle("Evaluation ({})".format(phase))
    plt.plot(targets.reshape(-1), label='targets')
    plt.legend()
    ax.set_title("Targets")
    ax.ticklabel_format(useOffset=False, style='plain')
    ax = fig.add_subplot(212)
    plt.plot(predictions.reshape(-1), label='predictions')
    plt.legend()
    ax.set_title("Prediction")
    ax.ticklabel_format(useOffset=False, style='plain')
    save_or_show_plot('eval_{}_target+predict'.format(phase))


def plot_metrics_comparison(title, snr, snr_label):
    if not visualize:
        return

    fig = plt.figure(figsize=FIG_SIZE_SINGLE)
    fig.add_subplot(111)
    fig.suptitle("{}".format(title))
    for i, r in enumerate(snr):
        plt.plot(r, label=snr_label[i], color=COLORS[i % len(COLORS)], marker='o')
    plt.legend()
    save_or_show_plot('eval_{}'.format(title))


def plot_line_from_tuples(values, scale, plot_name):
    if not visualize:
        return
    x = [r[0] for r in values]
    y = [r[1] for r in values]
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xscale(scale)
    save_or_show_plot(plot_name)


def plot_values(values, signal_labels, func_name, phase):
    if not visualize:
        return
    fig = plt.figure(figsize=FIG_SIZE_SINGLE)
    fig.add_subplot(111)
    fig.suptitle("{} CDF".format(func_name))
    for i, label in enumerate(signal_labels):
        x = values[i, :]
        x = x[np.logical_not(np.isnan(x))]
        x.sort()
        y = [v / len(x) for v in range(1, len(x)+1)]
        plt.plot(x, y, label=label, color=COLORS[i % len(COLORS)])
    plt.legend()
    save_or_show_plot('eval_{}_{}_cdf'.format(phase, func_name))


def plot_stat_from_tuples(value_tuples, plot_name, vertical_axis=False):
    if not visualize:
        return
    plt.subplots()

    categories = list(set([r[0] for r in value_tuples]))
    x = range(1, len(categories)+1)
    values = []
    for c in categories:
        cvalues = [v[1] for v in value_tuples if v[0] == c and v[1] is not None]
        values.append(cvalues)

    if vertical_axis:
        plt.boxplot(x, values)
        plt.xticks(x, categories, rotation='vertical')
    else:
        plt.boxplot(values, labels=categories)
    save_or_show_plot(plot_name)


def plot_rd_matrix_for_packet(targets, predictions, prediction_interf_substi, noisy_interfered, zero_substi_td, phase, packet_id, plot_substi, is_log_mag=False):
    if not visualize:
        return

    if task_id == 0:
        plot_rd_map(targets, "Evaluation ({}): Targets Doppler-Range Matrix".format(phase),
                    'eval_{}_doppler-range_matrix_targets_p{}'.format(phase, packet_id), is_log_mag)
        plot_rd_map(noisy_interfered, "Evaluation ({}): Noisy Doppler-Range Matrix".format(phase),
                    'eval_{}_doppler-range_matrix_interfered_p{}'.format(phase, packet_id), is_log_mag)
        plot_rd_map(zero_substi_td, "Evaluation ({}): Mitigation (zero substitude) Doppler-Range Matrix".format(phase),
                    'eval_{}_doppler-range_matrix_mitigation_zero_substi_p{}'.format(phase, packet_id), is_log_mag)

    plot_rd_map(predictions, "Evaluation ({}): Predictions Doppler-Range Matrix".format(phase),
                'eval_{}_doppler-range_matrix_predictions_p{}'.format(phase, packet_id), is_log_mag)

    if plot_substi:
        plot_rd_map(prediction_interf_substi,
                    "Evaluation ({}): Prediction (interference substitude) Doppler-Range Matrix".format(phase),
                    'eval_{}_doppler-range_matrix_predictions_substi_p{}'.format(phase, packet_id), is_log_mag)


def plot_target_range_doppler_matrix_with_and_out_interference(fft1_without_interference, fft1_with_interference, fft1_with_im_interference, fft1_with_re_interference):
    if task_id > 0:
        return

    data = fft1_without_interference
    plot_rd_map(data, "Range-Doppler Matrix without interference", 'targets_range-doppler_original')

    data = (fft1_without_interference - fft1_without_interference)
    plot_rd_map(data, "Range-Doppler Matrix without interference diff", 'targets_range-doppler_original_diff')

    # without interference
    data = fft1_with_interference
    plot_rd_map(data, "Range-Doppler Matrix with interference", 'targets_range-doppler_interference')

    data = (fft1_with_interference - fft1_without_interference)
    plot_rd_map(data, "Range-Doppler Matrix with interference diff", 'targets_range-doppler_interference_diff')

    # with imag interference
    data = fft1_with_im_interference
    plot_rd_map(data, "Range-Doppler Matrix with imag interference", 'targets_range-doppler_interference_imag')

    data = (fft1_with_im_interference - fft1_without_interference)
    plot_rd_map(data, "Range-Doppler Matrix with imag interference diff", 'targets_range-doppler_interference_imag_diff')

    # with real interference
    data = fft1_with_re_interference
    plot_rd_map(data, "Range-Doppler Matrix with real interference", 'targets_range-doppler_interference_re')

    data = (fft1_with_re_interference - fft1_without_interference)
    plot_rd_map(data, "Range-Doppler Matrix with real interference diff", 'targets_range-doppler_interference_re_diff')


def plot_rd_map(fft2, title, filename, is_log_mag=False):

    num_ramps = fft2.shape[1]

    v_vec_DFT_2 = v_vec_fft2(num_ramps)

    if not is_log_mag:
        fft2 = fft2 / np.amax(np.abs(fft2))
        fft2 = 10 * np.log10(np.abs(fft2))

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE_SINGLE)
    fig.suptitle(title)
    imgplot = plt.imshow(fft2, extent=[v_vec_DFT_2[0], v_vec_DFT_2[-1], 0, d_max], origin='lower', vmin=color_scale_min, vmax=color_scale_max)
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_aspect((v_vec_DFT_2[-1] - v_vec_DFT_2[0]) / d_max)
    plt.xlabel('velocity [m/s]')
    plt.ylabel('distance [m]')
    imgplot.set_cmap(STD_CMAP)
    plt.colorbar()
    save_or_show_plot(filename)


def plot_phase(rd_target, rd_test, phase, packet_id):
    if not visualize:
        return

    title = 'Phase comparison'
    filename = 'eval_{}_phase_p{}'.format(phase, packet_id)

    num_ramps = rd_target.shape[1]

    v_vec_DFT_2 = v_vec_fft2(num_ramps)

    rd_target_imag = np.imag(rd_target)
    rd_target_imag[np.logical_or(np.isnan(rd_target_imag), np.isinf(rd_target_imag))] = 0
    rd_target_real = np.real(rd_target)
    rd_target_real[np.logical_or(np.isnan(rd_target_real), np.isinf(rd_target_real))] = 0
    rd_phase_target = np.arctan(rd_target_imag.astype('float') / rd_target_real.astype('float'))

    rd_test_imag = np.imag(rd_test)
    rd_test_imag[np.logical_or(np.isnan(rd_test_imag), np.isinf(rd_test_imag))] = 0
    rd_test_real = np.real(rd_test)
    rd_test_real[np.logical_or(np.isnan(rd_test_real), np.isinf(rd_test_real))] = 0
    rd_phase_test = np.arctan(rd_test_imag.astype('float') / rd_test_real.astype('float'))

    rd_target_plot = rd_phase_target / np.amax(np.abs(rd_phase_target))
    rd_target_plot = 10 * np.log10(np.abs(rd_target_plot))

    rd_test_plot = rd_phase_test / np.amax(np.abs(rd_phase_test))
    rd_test_plot = 10 * np.log10(np.abs(rd_test_plot))

    phase_diff = rd_phase_target - rd_phase_test
    rd_diff_plot = phase_diff / np.amax(np.abs(phase_diff))
    rd_diff_plot = 10 * np.log10(np.abs(rd_diff_plot))

    fig = plt.figure(figsize=FIG_SIZE_HIGH)
    ax1 = fig.add_subplot(311)  # The big subplot
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    fig.suptitle(title)

    imgplot = ax1.imshow(rd_target_plot, extent=[v_vec_DFT_2[0], v_vec_DFT_2[-1], 0, d_max], origin='lower', vmin=color_scale_min, vmax=color_scale_max)
    ax1.set_title('Target')
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.set_aspect((v_vec_DFT_2[-1] - v_vec_DFT_2[0]) / d_max)
    ax1.set_xlabel('velocity [m/s]')
    ax1.set_ylabel('distance [m]')
    #imgplot.set_cmap(STD_CMAP)

    imgplot = ax2.imshow(rd_test_plot, extent=[v_vec_DFT_2[0], v_vec_DFT_2[-1], 0, d_max], origin='lower',
                        vmin=color_scale_min, vmax=color_scale_max)
    ax2.set_title('Prediction')
    ax2.ticklabel_format(useOffset=False, style='plain')
    ax2.set_aspect((v_vec_DFT_2[-1] - v_vec_DFT_2[0]) / d_max)
    ax2.set_xlabel('velocity [m/s]')
    ax2.set_ylabel('distance [m]')
    #imgplot.set_cmap(STD_CMAP)

    imgplot = ax3.imshow(rd_diff_plot, extent=[v_vec_DFT_2[0], v_vec_DFT_2[-1], 0, d_max], origin='lower',
                        vmin=color_scale_min, vmax=color_scale_max)
    ax3.set_title('Diff (T-P)')
    ax3.ticklabel_format(useOffset=False, style='plain')
    ax3.set_aspect((v_vec_DFT_2[-1] - v_vec_DFT_2[0]) / d_max)
    ax3.set_xlabel('velocity [m/s]')
    ax3.set_ylabel('distance [m]')
    #imgplot.set_cmap(STD_CMAP)

    #plt.colorbar()
    save_or_show_plot(filename)


def plot_object_mag_cuts(rd_denoised, rd_denoised_interf_substi, rd_clean,
                         rd_clean_noise, rd_interference, rd_zero_substi_td,
                         object_mask, packet_id, phase, is_rd, is_log_mag=False):
    if not is_log_mag:
        rd_clean = rd_clean / np.amax(np.abs(rd_clean))
        rd_clean = 10 * np.log10(np.abs(rd_clean))

        rd_denoised = rd_denoised / np.amax(np.abs(rd_denoised))
        rd_denoised = 10 * np.log10(np.abs(rd_denoised))

        rd_denoised_interf_substi = rd_denoised_interf_substi / np.amax(np.abs(rd_denoised_interf_substi))
        rd_denoised_interf_substi = 10 * np.log10(np.abs(rd_denoised_interf_substi))

        rd_clean_noise = rd_clean_noise / np.amax(np.abs(rd_clean_noise))
        rd_clean_noise = 10 * np.log10(np.abs(rd_clean_noise))

        rd_interference = rd_interference / np.amax(np.abs(rd_interference))
        rd_interference = 10 * np.log10(np.abs(rd_interference))

        rd_zero_substi_td = rd_zero_substi_td / np.amax(np.abs(rd_zero_substi_td))
        rd_zero_substi_td = 10 * np.log10(np.abs(rd_zero_substi_td))

    rd_log_mag_signals = [rd_clean, rd_clean_noise, rd_interference, rd_zero_substi_td, rd_denoised, rd_denoised_interf_substi]
    signal_labels = ['Clean', 'Noisy+C', 'Interference+C+N', 'Mitigation: Zero Substitude TD', 'Denoised', 'Denoised Interference Substitude']

    rows, columns = np.nonzero(object_mask)

    for i in range(min(len(rows), 3)):
        r = rows[i]
        c = columns[i]
        plot_row_column_cuts(rd_log_mag_signals, signal_labels, c, r, i, packet_id, phase, is_rd, is_mag=True)


def plot_object_phase_cuts(rd_denoised, rd_denoised_interf_substi, rd_clean,
                           rd_clean_noise, rd_interference, rd_zero_substi_td, object_mask,
                           packet_id, phase, is_rd):

    rd_phase_clean = phase_by_rd(rd_clean)
    rd_phase_denoised = phase_by_rd(rd_denoised)
    rd_phase_clean_noise = phase_by_rd(rd_clean_noise)
    rd_phase_interference = phase_by_rd(rd_interference)
    rd_phase_zero_substi_td = phase_by_rd(rd_zero_substi_td)
    rd_phase_denoised_substi = phase_by_rd(rd_denoised_interf_substi)

    rd_log_mag_signals = [rd_phase_clean, rd_phase_clean_noise, rd_phase_interference, rd_phase_zero_substi_td, rd_phase_denoised, rd_phase_denoised_substi]
    signal_labels = ['Clean', 'Noisy+C', 'Interference+C+N', 'Mitigation: Zero Substitude TD', 'Denoised', 'Denoised Interference Substitude']

    rows, columns = np.nonzero(object_mask)

    for i in range(min(len(rows), 3)):
        r = rows[i]
        c = columns[i]
        plot_row_column_cuts(rd_log_mag_signals, signal_labels, c, r, i, packet_id, phase, is_rd, is_mag=False)


def phase_by_rd(rd):
    rd_imag = np.imag(rd)
    rd_imag[np.logical_or(np.isnan(rd_imag), np.isinf(rd_imag))] = 0
    rd_real = np.real(rd)
    rd_real[np.logical_or(np.isnan(rd_real), np.isinf(rd_real))] = 0
    rd_phase = np.arctan(rd_imag.astype('float') / rd_real.astype('float'))
    return rd_phase


def plot_row_column_cuts(rd_log_mag_signals, signal_labels, obj_col, obj_row, obj_id, packet_id, phase, is_rd, is_mag):
    shape0 = len(rd_log_mag_signals[0][:, obj_col])
    if is_rd:
        x_label_row = 'velocity [m/s]'
        row_title = 'Velocity cut'
        filename_add1 = 'rd'
        shape1 = len(rd_log_mag_signals[0][obj_row, :])
        x_vec2 = v_vec_fft2(shape1)
        x_vec1 = np.array(np.linspace(0, 1, shape0)) * d_max
    else:
        x_label_row = 'cross range [m]'
        row_title = 'Cross Range cut'
        filename_add1 = 'aoa'
        x_vec2, x_vec1 = basis_vec_fft3()
        x_vec2 = x_vec2[obj_row, :]
        x_vec1 = x_vec1[:, obj_col]

    if is_mag:
        y_label = 'log mag'
        filename_add2 = 'mag'
    else:
        y_label = 'phase'
        filename_add2 = 'phase'

    x_label_col = 'range [m]'

    #  Column cut
    fig = plt.figure(figsize=FIG_SIZE)
    fig.suptitle("Object cuts ({} - p={} o={})".format(phase, packet_id, obj_id))

    ax = fig.add_subplot(211)

    for i in range(len(rd_log_mag_signals)):
        signal = rd_log_mag_signals[i][:, obj_col]
        plt.plot(x_vec1, signal, label="{}".format(signal_labels[i]))

    plt.axvline(x=x_vec1[obj_row])

    #  Row cut
    ax.set_title('Range cut')
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.legend()
    plt.xlabel(x_label_col)
    plt.ylabel(y_label)

    ax = fig.add_subplot(212)
    for i in range(len(rd_log_mag_signals)):
        signal = rd_log_mag_signals[i][obj_row, :]
        plt.plot(x_vec2, signal, label="{}".format(signal_labels[i]))
    plt.axvline(x=x_vec2[obj_col])

    ax.set_title(row_title)
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.legend()
    plt.xlabel(x_label_row)
    plt.ylabel(y_label)

    save_or_show_plot('eval_{}_{}_object_cut_{}_p{}_o{}'.format(phase, filename_add1, filename_add2, packet_id, obj_id))


def plot_phase_amplitude_for_packet(targets, predictions, object_mask, phase, packet_id):
    if not visualize:
        return

    targets = targets.transpose()
    predictions = predictions.transpose()

    rows, columns = np.nonzero(object_mask)

    for i in range(min(len(rows), 1)):
        fts_idx = rows[i]
        if task_id == 0:
            plot_phase_amplitude_for_fts(targets[fts_idx, :], fts_idx, phase, packet_id, 'target')

        plot_phase_amplitude_for_fts(predictions[fts_idx, :], fts_idx, phase, packet_id, 'denoised')

    fts_idx = 20
    if task_id == 0:
        plot_phase_amplitude_for_fts(targets[fts_idx, :], fts_idx, phase, packet_id, 'target_no_obj')

    plot_phase_amplitude_for_fts(predictions[fts_idx, :], fts_idx, phase, packet_id, 'denoised_no_obj')


def plot_phase_amplitude_for_fts(fft1_data, fts_idx, eval_phase, packet_idx, title_add):
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE_SINGLE)
    fig.suptitle('Phase & Amplitude FFT1 ({} {}: p={}, fts={})'.format(eval_phase, title_add, packet_idx, fts_idx))

    ax = fig.add_subplot(211)
    phase = np.arctan(np.imag(fft1_data) / np.real(fft1_data))
    plt.plot(phase, label="phase")
    ax.set_title('Phase')
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.xlabel('velocity [m/s]')
    plt.legend()

    ax = fig.add_subplot(212)
    magnitude = 10 * np.log10(np.abs(fft1_data))
    plt.plot(magnitude, label="magnitude")
    ax.set_title('Magnitude')
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.xlabel('velocity [m/s]')
    plt.legend()

    save_or_show_plot('eval_{}_phase_ampli_{}_p{}_fts{}'.format(eval_phase, title_add, packet_idx, fts_idx))


def plot_distance_map(fft1_data, title, filename):

    num_ramps = fft1_data.shape[1]

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE_SINGLE)
    fig.suptitle(title)
    imgplot = plt.imshow(np.abs(fft1_data), extent=[0, num_ramps, 0, d_max], origin='lower')
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_aspect(num_ramps / d_max)
    plt.xlabel('ramps')
    plt.ylabel('distance [m]')
    imgplot.set_cmap(STD_CMAP)
    plt.colorbar()
    save_or_show_plot(filename)


def plot_rd_noise_mask(noise_mask, title, filename):
    num_ramps = noise_mask.shape[1]
    v_vec_DFT_2 = v_vec_fft2(num_ramps)

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE_SINGLE)
    fig.suptitle(title)
    imgplot = plt.imshow(noise_mask.astype(int), extent=[v_vec_DFT_2[0], v_vec_DFT_2[-1], 0, d_max], origin='lower')
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_aspect((v_vec_DFT_2[-1] - v_vec_DFT_2[0]) / d_max)
    plt.xlabel('velocity [m/s]')
    plt.ylabel('distance [m]')
    imgplot.set_cmap(STD_CMAP)
    plt.colorbar()
    save_or_show_plot(filename)


def plot_aoa_noise_mask(noise_mask, title, filename):
    x_vec, y_vec = basis_vec_fft3()

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.gca(projection='3d')

    fig.suptitle(title)
    surf = ax.plot_surface(x_vec, y_vec, noise_mask.astype(int), cmap=STD_CMAP, linewidth=0, vmin=0, vmax=1, rstride=1, cstride=1)

    plt.xlabel('cross range [m]')
    plt.ylabel('range [m]')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.view_init(87.5, -90)
    plt.draw()

    save_or_show_plot(filename)


def plot_angle_of_arrival_map(fft3, title, filename):
    x_vec, y_vec = basis_vec_fft3()
    fft3_plot = 10 * np.log10(np.abs(fft3 / np.amax(np.abs(fft3))))

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.gca(projection='3d')
    fig.suptitle(title)
    surf = ax.plot_surface(x_vec, y_vec, fft3_plot, cmap=STD_CMAP, linewidth=0, vmin=color_scale_min, vmax=color_scale_max, rstride=1, cstride=1)

    plt.xlabel('cross range [m]')
    plt.ylabel('range [m]')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.view_init(87.5, -90)
    plt.draw()

    save_or_show_plot(filename)


def plot_cross_ranges(obj_idx, rows, columns, phase, scene_idx, x_vec,
                      angular_spectrum, angular_spectrum_clean, angular_spectrum_interf,
                      angular_spectrum_original, angular_spectrum_zero_substi, object_mask):
    cr_prediction = calculate_cross_range_fft(angular_spectrum[rows[obj_idx], columns[obj_idx]])
    cr_clean = calculate_cross_range_fft(angular_spectrum_clean[rows[obj_idx], columns[obj_idx]])
    cr_interf = calculate_cross_range_fft(angular_spectrum_interf[rows[obj_idx], columns[obj_idx]])
    cr_original = calculate_cross_range_fft(angular_spectrum_original[rows[obj_idx], columns[obj_idx]])
    cr_zero_substi = calculate_cross_range_fft(angular_spectrum_zero_substi[rows[obj_idx], columns[obj_idx]])

    signals = [cr_clean, cr_prediction, cr_interf, cr_original, cr_zero_substi]
    labels = ['clean', 'prediction', 'interf', 'original', 'zero substi']

    fig = plt.figure(figsize=FIG_SIZE)
    fig.suptitle('Cross range (s{})'.format(scene_idx))
    fig.add_subplot(111)
    for i in range(len(signals)):
        s = signals[i]
        s = 10 * np.log10(np.abs(s / np.amax(np.abs(s))))
        plt.plot(x_vec, s[0], label=labels[i])

    _, obj_indices = np.nonzero(object_mask)
    for i in obj_indices:
        plt.axvline(x=x_vec[i])
    plt.xlabel('cross range [m]')
    plt.ylabel('log mag')
    plt.legend()

    save_or_show_plot('eval_{}_cr_s{}_o{}'.format(phase, scene_idx, obj_idx))
