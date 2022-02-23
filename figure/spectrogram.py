"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/2/23 下午9:50
@Author  : Yang "Jan" Xiao 
@Description : spectrogram
"""

import librosa.display
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from audiomentations import AddGaussianNoise, PitchShift, Shift, FrequencyMask, ClippingDistortion, TimeMask

path = "../test/test.wav"
pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
matplotlib.rcParams.update(pgf_with_rc_fonts)


def plot_spectrogram(data, fs, name):
    L = len(data)
    print('Time:', L / fs)

    # 归一化
    data = data * 1.0 / max(data)

    # 0.025s
    framelength = 0.025
    # NFFT点数=0.025*fs
    framesize = int(framelength * fs)
    print("NFFT:", framesize)

    # 提取mel特征
    mel_spect = librosa.feature.melspectrogram(data, sr=fs, n_fft=framesize)
    # 转化为log形式
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    # 画mel谱图
    # set_style()
    librosa.display.specshow(mel_spect, sr=fs, x_axis='time', y_axis='mel')
    plt.ylabel('Mel Frequency')
    plt.xlabel('Time(s)')
    plt.title('Mel Spectrogram')
    plt.savefig(name + ".pdf")
    plt.show()


if __name__ == "__main__":
    data, fs = librosa.load(path, sr=None, mono=False)
    shift = Shift(min_fraction=-0.5, max_fraction=0.5, p=1)
    timemask = TimeMask(min_band_part=0.3, max_band_part=0.5, p=1)
    freqmask = FrequencyMask(min_frequency_band=0.1, max_frequency_band=0.3, p=1)
    clip = ClippingDistortion(min_percentile_threshold=10, max_percentile_threshold=30, p=1)
    addnoise = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1)
    pitchshift = PitchShift(min_semitones=-10, max_semitones=10, p=1)
    shift_data = shift(data, sample_rate=fs)
    timemask_data = timemask(data, sample_rate=fs)
    clip_data = clip(data, sample_rate=fs)
    freqmask_data = freqmask(data, sample_rate=fs)
    addnoise_data = addnoise(data, sample_rate=fs)
    pitchshift_data = pitchshift(data, sample_rate=fs)
    plot_spectrogram(shift_data, fs, name="shift")
    plot_spectrogram(timemask_data, fs, name="timemask")
    plot_spectrogram(clip_data, fs, name="clip")
    plot_spectrogram(freqmask_data, fs, name="freqmask")
    plot_spectrogram(addnoise_data, fs, name="addnoise")
    plot_spectrogram(pitchshift_data, fs, name="pitchshift")
