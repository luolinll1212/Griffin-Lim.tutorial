# -*- coding: utf-8 -*-
import numpy as np
from utils import audio
from hparams import hparams as hps


path = r'./data/000001.wav'

# 第一步，加载语音，数据本来就是[-1,1]，所以不需要归一化
wav = audio.load_wav(path, hps.sample_rate)

# 第二步，去除前后的静音
if hps.trim_silence:
    wav = audio.trim_silence(wav, hps)

# 第三步，计算mel图谱
mel_spectrogram = audio.melspectrogram(wav, hps).astype(np.float32)

#第四步，计算线性图谱(声谱图)
linear_spectrogram = audio.linearspectrogram(wav, hps).astype(np.float32)

savename = path.split('/')[-1].split('.')[0]
mel_filename = './data/mel-{}.npy'.format(savename)
linear_filename = './data/linear-{}.npy'.format(savename)

np.save(mel_filename, mel_spectrogram.T, allow_pickle=False)
np.save(linear_filename, linear_spectrogram.T, allow_pickle=False)