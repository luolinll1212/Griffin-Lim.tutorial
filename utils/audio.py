# -*- coding: utf-8 -*-
import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, hps):
    wav = wav / np.abs(wav).max() * 0.999
    f1 = 0.5 * 32767 / max(0.01, np.max(np.abs(wav)))
    f2 = np.sign(wav) * np.power(np.abs(wav), 0.95)
    wav = f1 * f2
    wav = signal.convolve(wav, signal.firwin(hps.num_freq, [hps.fmin, hps.fmax], pass_zero=False, fs=hps.sample_rate))
    # proposed by @dsmiller
    wavfile.write(path, hps.sample_rate, wav.astype(np.int16))

def trim_silence(wav, hps):
    return librosa.effects.trim(wav, top_db=hps.trim_top_db, frame_length=hps.trim_fft_size,
                                hop_length=hps.trim_hop_size)[0]
def preemphasis(wav, k):
    return signal.lfilter([1, -k], [1], wav)

def linearspectrogram(wav, hps):
    if hps.is_preemphasis:
        wav = preemphasis(wav, hps.preemphasis)
    D = _stft(wav, hps)
    S = _amp_to_db(np.abs(D), hps) - hps.ref_level_db

    if hps.signal_normalization:
        return _normalize(S, hps)
    return S

def melspectrogram(wav, hps):
    if hps.is_preemphasis:
        wav = preemphasis(wav, hps.preemphasis)
    D = _stft(wav, hps)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hps), hps) - hps.ref_level_db
    if hps.signal_normalization:
        return _normalize(S, hps)
    return S

def _stft(y, hps):
    return librosa.stft(y=y, n_fft=hps.n_fft, hop_length=get_hop_size(hps), win_length=hps.win_size)

def get_hop_size(hps):
    hop_size = hps.hop_size
    if hop_size is None:
        assert hps.frame_shift_ms is not None
        hop_size = int(hps.frame_shift_ms / 1000 * hps.sample_rate)
    return hop_size

# Conversions
_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectogram, hps):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hps)
    return np.dot(_mel_basis, spectogram)

def _mel_to_linear(mel_spectrogram, hps):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hps))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _build_mel_basis(hps):
    assert hps.fmax <= hps.sample_rate // 2
    return librosa.filters.mel(hps.sample_rate, hps.n_fft, n_mels=hps.num_mels,
                               fmin=hps.fmin, fmax=hps.fmax)

def _amp_to_db(x, hps):
    min_level = np.exp(hps.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _normalize(S, hps):
    if hps.allow_clipping_in_normalization:
        if hps.symmetric_mels:
            return np.clip((2 * hps.max_abs_value) * (
                (S - hps.min_level_db) / (-hps.min_level_db)) - hps.max_abs_value,
                           -hps.max_abs_value, hps.max_abs_value)
        else:
            return np.clip(hps.max_abs_value * ((S - hps.min_level_db) / (-hps.min_level_db)), 0,
                           hps.max_abs_value)

    if hps.symmetric_mels:
        return (2 * hps.max_abs_value) * ((S - hps.min_level_db) / (-hps.min_level_db)) - hps.max_abs_value
    else:
        return hps.max_abs_value * ((S - hps.min_level_db) / (-hps.min_level_db))

def inv_linear_spectrogram(linear_spectrogram, hps):
    '''Converts linear spectrogram to waveform using librosa'''
    if hps.signal_normalization:
        D = _denormalize(linear_spectrogram, hps)
    else:
        D = linear_spectrogram

    S = _db_to_amp(D + hps.ref_level_db)  # Convert back to linear

    if hps.is_preemphasis:
        return inv_preemphasis(_griffin_lim(S ** hps.power, hps), hps.preemphasis)

    return _griffin_lim(S ** hps.power, hps)

def inv_mel_spectrogram(mel_spectrogram, hps):
    '''Converts mel spectrogram to waveform using librosa'''
    if hps.signal_normalization:
        D = _denormalize(mel_spectrogram, hps)
    else:
        D = mel_spectrogram

    S = _mel_to_linear(_db_to_amp(D + hps.ref_level_db), hps)  # Convert back to linear

    if hps.is_preemphasis:
        return inv_preemphasis(_griffin_lim(S ** hps.power, hps), hps.preemphasis)

    return _griffin_lim(S ** hps.power, hps)

def _denormalize(D, hps):
    if hps.allow_clipping_in_normalization:
        if hps.symmetric_mels:
            return (((np.clip(D, -hps.max_abs_value,
                              hps.max_abs_value) + hps.max_abs_value) * -hps.min_level_db / (2 * hps.max_abs_value))
                    + hps.min_level_db)
        else:
            return ((np.clip(D, 0, hps.max_abs_value) * -hps.min_level_db / hps.max_abs_value) + hps.min_level_db)

    if hps.symmetric_mels:
        return (((D + hps.max_abs_value) * -hps.min_level_db / (2 * hps.max_abs_value)) + hps.min_level_db)
    else:
        return ((D * -hps.min_level_db / hps.max_abs_value) + hps.min_level_db)

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def inv_preemphasis(wav, k):
    return signal.lfilter([1], [1, -k], wav)

def _istft(y, hps):
    return librosa.istft(y, hop_length=get_hop_size(hps), win_length=hps.win_size)

def _griffin_lim(S, hps):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hps)
    for i in range(hps.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, hps)))
        y = _istft(S_complex * angles, hps)
    return y

