# -*- coding: utf-8 -*-  
# -*- coding: utf-8 -*-

# 定义参数
class hparams:

    ################################
    # 音频处理                     #
    ################################
    sample_rate = 48_000

    # Matlab(或其他数据集) 调整参数
    trim_silence = True
    trim_fft_size = 512
    trim_hop_size = 128
    trim_top_db = 60

    # FIR滤波器
    is_preemphasis = False
    preemphasis = 0.97 # 预修正参数

    # 信号分析，分帧参数
    n_fft = 4096            # 傅里叶变换的窗口
    win_size = 2400         # 分帧的窗口
    hop_size = 600          # 分帧的帧移，所以1秒(48k个点)，帧移 = (48k * 12.5) / 1k = 600
    frame_shift_ms = 12.5   # 1秒(1k个点)，帧移12.5，经验值

    # mel滤波器上下限制
    num_mels = 160
    num_freq = 2049  # (= n_fft / 2 + 1) 傅里叶变换是双边，只去前一半部分
    fmax = 7600
    fmin = 125
    min_level_db = -120
    ref_level_db = 20

    # mel和线性图谱(声谱图)的标准化/缩放和剪切
    signal_normalization = True
    allow_clipping_in_normalization = False #仅当mel_normalization = True时才相关
    symmetric_mels = True
    max_abs_value = 4.

    #Griffin Lim算法参数
    power = 1.2
    griffin_lim_iters = 60



