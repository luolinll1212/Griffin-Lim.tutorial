# -*- coding: utf-8 -*-  
# -*- coding: utf-8 -*-
import os
import numpy as np
from utils import audio
from hparams import hparams as hps


linear_path =  './data/linear-000001.npy'
linear_name = linear_path.split('/')[-1].split('.')[0]
linear_p = np.load(linear_path)

mel_path = r'./data/mel-000001.npy'
mel_name = mel_path.split('/')[-1].split('.')[0]
mel_p = np.load(mel_path)


# 保存线性频谱
wav = audio.inv_linear_spectrogram(linear_p.T, hps)
audio.save_wav(wav, os.path.join("./data","{}.wav".format(linear_name)), hps)

# 保存mel频谱
wav = audio.inv_mel_spectrogram(mel_p.T, hps)
audio.save_wav(wav, os.path.join("./data","{}.wav".format(mel_name)), hps)