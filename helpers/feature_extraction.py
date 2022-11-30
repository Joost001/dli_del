"""Contains common feature extraction tools.

Spectrogram: shows the frequency make-up of an audio signal as it varies with time.
Mel Filter Bank: the filter bank for converting frequency bins to mel-scale bins.
Mel-Scale Spectrogram: Generating a mel-scale spectrogram involves generating a spectrogram and performing mel-scale conversion
MFCC: Apply Mel-Scale Spectrogram to speech_waveform, filter bank, and then DCT
"""
import librosa
import torch
import torchaudio.transforms as T
import matplotlib.pyplot as plt


torch.random.manual_seed(0)


def plot_waveform(waveform, sr, title="Waveform"):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr
    figure, axes = plt.subplots(num_channels, 1)
    axes.plot(time_axis, waveform[0], linewidth=1)
    axes.grid(True)
    figure.suptitle(title)
    plt.show(block=True)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram),
                    origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=True)


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")
    plt.show(block=False)


def mel_spectrogram(speech_waveform, kwargs=None):
    """Create MelSpectrogram for a raw audio signal.
    
    :param speech_waveform (torch.Tensor): output form torchaudio.load()[0] 
    :param sample_rate (int, optional): Sample rate of audio signal. (Default: 16000)
    :param n_fft (int, optional): Size of FFT, creates n_fft // 2 + 1 bins. (Default: 400) 
    :param win_length (int or None, optional): Window size. (Default: n_fft)
    :param hop_length (int or None, optional): Length of hop between STFT windows. (Default: win_length // 2)
    :param f_min (float, optional): Minimum frequency. (Default: 0.)
    :param f_max (float or None, optional): Maximum frequency. (Default: None)
    :param pad (int, optional): Two sided padding of signal. (Default: 0)
    :param n_mels (int, optional):  Number of mel filterbanks. (Default: 128)
    :param window_fn (Callable[..., Tensor], optional): A function to create a window tensor that is applied/multiplied to each frame/window. (Default: torch.hann_window)
    :param power (float, optional): Exponent for the magnitude spectrogram, (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: 2)
    :param normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: False)
    :param wkwargs (Dict[..., ...] or None, optional): Arguments for window function. (Default: None)
    :param center (bool, optional): whether to pad waveform on both sides so that the t-th frame is centered at time t X hop_length. (Default: True)
    :param pad_mode (string, optional): controls the padding method used when center is True. (Default: "reflect")
    :param onesided (bool, optional): controls whether to return half of results to avoid redundancy. (Default: True)
    :param norm (str or None, optional): If “slaney”, divide the triangular mel weights by the width of the mel band (area normalization). (Default: None)
    :param mel_scale (str, optional): Scale to use: htk or slaney. (Default: htk)
    """
    mel_spectrogram = T.MelSpectrogram(**kwargs)
    melspec = mel_spectrogram(speech_waveform)
    return melspec


def mfcc_transform(speech_waveform, kwargs=None):
    """Create the Mel-frequency cepstrum coefficients from an audio signal. By default, this calculates the MFCC on the 
    DB-scaled Mel spectrogram. This is not the textbook implementation, but is implemented here to give consistency 
    with librosa. This output depends on the maximum value in the input spectrogram, and so may return different values 
    for an audio clip split into snippets vs. a a full clip.

    :param speech_waveform (torch.Tensor): output form torchaudio.load()[0]
    :param sample_rate (int, optional): Sample rate of audio signal. (Default: 16000)
    :param n_mfcc (int, optional): Number of mfc coefficients to retain. (Default: 40)
    :param dct_type (int, optional): type of DCT (discrete cosine transform) to use. (Default: 2)
    :param norm (str, optional): norm to use. (Default: "ortho")
    :param log_mels (bool, optional): whether to use log-mel spectrograms instead of db-scaled. (Default: False)
    :param melkwargs (dict or None, optional): arguments for MelSpectrogram. (Default: None)
    """
    transform = T.MFCC(**kwargs) if kwargs else T.MFCC()
    return transform(speech_waveform)

