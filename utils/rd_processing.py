import numpy as np

v_max = 20.545205434019756
d_max = 1.534204175895034e+02
num_angle_fft_bins = 1024

basis_x_fft3 = None
basis_y_fft3 = None
hann_fft1 = None
hann_fft2 = None
hann_fft3 = None
v_vec = None
d_vec = None


def calculate_range_fft(s_IF, antenna):
    total_num_ramps = s_IF.shape[0]
    num_fts = s_IF.shape[1]

    if len(s_IF.shape) == 3:
        s_IF = s_IF[:, :, antenna]

    hann = hanning_window_fft1(num_fts, total_num_ramps)
    sig_win = s_IF * hann
    fft = 1 / np.sqrt(num_fts) * np.fft.fft(sig_win, num_fts, axis=1)[:, 0:int(num_fts / 2)]
    return fft


def calculate_velocity_fft(fft1):
    num_ramps = fft1.shape[0]
    num_fts = fft1.shape[1]

    fft1 = fft1.transpose()

    hann = hanning_window_fft2(num_fts, num_ramps)
    sig_win = fft1 * hann
    rd = 1 / np.sqrt(num_ramps) * np.fft.fftshift(np.fft.fft(sig_win, axis=1), axes=1)
    return rd


def calculate_angle_fft(fft1):
    num_samples = fft1.shape[0]
    num_antennas = fft1.shape[1]

    hann = hanning_window_fft3(num_samples, num_antennas)
    sig_win = fft1 * hann
    aoa = np.fft.fftshift(np.fft.fft(sig_win, num_angle_fft_bins, axis=1), axes=1)
    return aoa


def calculate_cross_range_fft(obj_peaks_ever_antennas):
    num_antennas = len(obj_peaks_ever_antennas)

    hann = hanning_window_fft3(1, num_antennas)
    sig_win = obj_peaks_ever_antennas * hann
    cr = np.fft.fftshift(np.fft.fft(sig_win, num_angle_fft_bins, axis=1), axes=1)
    return cr


def hanning_window_fft1(num_fts, num_ramps):
    global hann_fft1
    if hann_fft1 is not None:
        return hann_fft1
    hann1 = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, num_fts + 1) / (num_fts - 1)))
    hann = np.matlib.repmat(hann1, num_ramps, 1)

    hann_fft1 = hann
    return hann


def hanning_window_fft2(num_fts, num_ramps):
    global hann_fft2
    if hann_fft2 is not None:
        return hann_fft2
    hann1 = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, num_ramps + 1) / (num_ramps - 1)))
    hann = np.matlib.repmat(hann1, num_fts, 1)

    hann_fft2 = hann
    return hann


def hanning_window_fft3(num_samples, num_antennas):
    global hann_fft3
    if hann_fft3 is not None and hann_fft3.shape[0] == num_samples and hann_fft3.shape[1] == num_antennas:
        return hann_fft3
    hann1 = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, num_antennas + 1) / (num_antennas - 1)))
    hann = np.matlib.repmat(hann1, num_samples, 1)

    hann_fft3 = hann
    return hann


def d_vec_fft2(num_fts):
    global d_vec
    if d_vec is not None and len(d_vec) == num_fts:
        return d_vec
    d_vec = np.array(np.linspace(0, 1, num_fts)) * d_max
    return d_vec


def v_vec_fft2(num_ramps):
    global v_vec
    if v_vec is not None and len(v_vec) == num_ramps:
        return v_vec
    v_vec = np.arange(-num_ramps / 2, num_ramps / 2, 1).transpose() / num_ramps * 2 * v_max
    return v_vec


def basis_vec_fft3():
    global basis_x_fft3, basis_y_fft3
    if basis_x_fft3 is not None and basis_y_fft3 is not None:
        return basis_x_fft3, basis_y_fft3

    a_vec = np.arcsin(1 * np.linspace(-1, 1, num_angle_fft_bins))
    d_vec = np.linspace(0, 1, num_angle_fft_bins) * d_max

    basis_x = np.zeros((len(d_vec), len(a_vec)))
    basis_y = np.zeros((len(d_vec), len(a_vec)))
    for i_dist in range(len(d_vec)):
        basis_x[i_dist, :] = d_vec[i_dist] * np.cos(a_vec + np.pi / 2)
        basis_y[i_dist, :] = d_vec[i_dist] * np.sin(a_vec + np.pi / 2)

    basis_x_fft3 = basis_x
    basis_y_fft3 = basis_y

    return basis_x, basis_y


def basis_vec_cross_range(d_idx):
    a_vec = np.arcsin(1 * np.linspace(-1, 1, num_angle_fft_bins))
    d_vec = np.linspace(0, 1, num_angle_fft_bins) * d_max

    basis_vec = d_vec[d_idx] * np.cos(a_vec + np.pi / 2)

    return basis_vec
