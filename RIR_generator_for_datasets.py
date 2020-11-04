#!/usr/bin/env python
# coding: utf-8

# # Room Impulse Response Generator

# In[9]:


import wave as wave
import pyroomacoustics as pa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

import soundfile as sf

import IPython.display


# In[10]:


np.random.seed(0)
# 畳み込みに用いる音声波形
clean_wave_file = "../data/NoisySpeechDatabase/noisy_trainset_28spk_wav_16kHz/p230_013.wav"

wav = wave.open(clean_wave_file)
data = wav.readframes(wav.getnframes())
data = np.frombuffer(data, dtype=np.int16)
data = data/np.iinfo(np.int16).max
wav.close()

# 音声再生
IPython.display.Audio(clean_wave_file)


# #### Pyroomacousticsを用いた室内伝達関数（RIR）のシミュレーションと畳み込み

# In[56]:


# 指定するパラメータ
# 畳み込みに用いる波形
clean_wave_files = ["../data/NoisySpeechDatabase/noisy_trainset_28spk_wav_16kHz/p230_013.wav"]
# 畳み込み後の波形の保存先パス
save_path = "./wav/convolved.wav"
# サンプリング周波数
sample_rate = 16000
# 音声と雑音の比率 [dB]
SNR = 90.
# 音源とマイクロホンの距離 [m]
distance_mic_to_source=2. 
# 音源方向（音源が複数ある場合はリストに追加）
azimuth = [0] # 方位角
elevation = [np.pi/6] # 仰角
# 部屋（シミュレーション環境）の設定
room_width = 5.0
room_length = 5.0
room_height = 5.0
# 部屋の残響を設定
max_order = 30 #　部屋の壁で何回音が反射するか（反射しない場合0）
absorption = 0.2 # 部屋の壁でどの程度音が吸収されるか （吸収されない場合None）

# 以下は固定
# 部屋の３次元形状を表す（単位はm）
room_dim = np.r_[room_width, room_length, room_height]
print("部屋の3次元形状：", room_dim)

# マイクロホンアレイの中心位置
nakbot_height = 0.57 # Nakbotの全長
mic_array_height = nakbot_height - 0.04 # 0.04はTAMAGO-03マイクロホンアレイの頂上部からマイクロホンアレイ中心までの距離
mic_array_loc = np.r_[room_width/2, room_length/2, 0] + [0, 0, mic_array_height] # 部屋の中央に配置されたNakbot上のマイクロホンアレイ
print("マイクロホンアレイ中心座標：", mic_array_loc)
# TAMAGO-03のマイクロホンアレイのマイクロホン配置（単位はm）
mic_alignments = np.array(
[
    [0.035, 0.0, 0.0],
    [0.035/np.sqrt(2), 0.035/np.sqrt(2), 0.0],
    [0.0, 0.035, 0.0],
    [-0.035/np.sqrt(2), 0.035/np.sqrt(2), 0.0],
    [-0.035, 0.0, 0.0],
    [-0.035/np.sqrt(2), -0.035/np.sqrt(2), 0.0],
    [0.0, -0.035, 0.0],
    [0.035/np.sqrt(2), -0.035/np.sqrt(2), 0.0]
])
n_channels = np.shape(mic_alignments)[0]
print("マイクロホン数：", n_channels)
# get the microphone array　（各マイクロホンの空間的な座標）
R = mic_alignments.T + mic_array_loc[:, None]
"""R: (3D coordinates [m], num_microphones)"""

# 音源の位置（HARK座標系に対応） [仰角θ, 方位角φ]
doas = np.array(
[[elevation[0], azimuth[0]], # １個目の音源 
# [elevation[1], azimuth[1]] # ２個目の音源
])
n_sources = len(clean_wave_files)
print("音源数:", n_sources)
source_locations = np.zeros((3, doas.shape[0]), dtype=doas.dtype)
"""source_locations: (xyz, num_sources)"""
source_locations[0,  :] = np.cos(doas[:, 1]) * np.cos(doas[:, 0]) # x = rcosφcosθ
source_locations[1,  :] = np.sin(doas[:, 1]) * np.cos(doas[:, 0]) # y = rsinφcosθ
source_locations[2,  :] = np.sin(doas[:, 0]) # z = rsinθ
source_locations *= distance_mic_to_source
source_locations += mic_array_loc[:, None] # マイクロホンアレイからの相対位置→絶対位置
for i in range(n_sources):
    x = source_locations[0, i]
    y = source_locations[1, i]
    z = source_locations[2, i]
    print("{}個目の音源の位置： (x, y, z) = ({}, {}, {})".format(i+1, x, y, z))
    
# 音声波形の長さを調べる
n_samples = 0
# ファイルを読み込む
for clean_wave_file in clean_wave_files:
    wav = wave.open(clean_wave_file)
    if n_samples<wav.getnframes():
        n_samples=wav.getnframes()
    wav.close()
clean_data = np.zeros([n_sources, n_samples])
# ファイルを読み込む
s = 0
for clean_wave_file in clean_wave_files:
    wav = wave.open(clean_wave_file)
    data = wav.readframes(wav.getnframes())
    data = np.frombuffer(data, dtype=np.int16)
    data = data/np.iinfo(np.int16).max
    clean_data[s, :wav.getnframes()] = data
    wav.close()
    s = s+1

# 部屋を生成する
room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=max_order, absorption=absorption)
#　用いるマイクロホンアレイの情報を設定する
room.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))
# 各音源をシミュレーションに追加する
for s in range(n_sources):
    clean_data[s] /= np.std(clean_data[s])
    room.add_source(source_locations[:, s], signal=clean_data[s])
# RIRのシミュレーション生成と音源信号への畳み込みを実行
room.simulate(snr=SNR)
convolved_wave = room.mic_array.signals.T/np.max(room.mic_array.signals.T)
print("畳み込み後の音声波形:", convolved_wave.shape)

# インパルス応答の取得と残響時間（RT60）の取得
impulse_responses = room.rir
rt60 = pa.experimental.measure_rt60(impulse_responses[0][0], fs=sample_rate)
print("残響時間:{} [sec]".format(rt60))

# 音声保存
sf.write(save_path, convolved_wave, sample_rate)
# 音声再生
IPython.display.Audio(save_path)


# In[71]:


# スペクトログラムを表示
# 短時間フーリエ変換
f, t, stft_data = signal.stft(room.mic_array.signals, fs=wav.getframerate(), window="hann", nperseg=512, noverlap=256)

# スペクトログラムを表示する
fig = plt.figure(figsize=(10, 4))
spectrogram, freqs, t, im = plt.specgram(data, NFFT=512, noverlap=512/16*15, Fs=wav.getframerate(), cmap="gray")

# カラーバーを表示する
fig.colorbar(im).set_label('Intensity [dB]')

# x軸のラベル
plt.xlabel("Time [sec]")

# y軸のラベル
plt.ylabel("Frequency [Hz]")

# 音声ファイルを画像として保存
plt.savefig("./img/spectrogram.png")

# 画像を画面に表示
plt.show()


# In[ ]:




