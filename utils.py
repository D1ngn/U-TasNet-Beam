import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

from scipy import signal


# 音声データをロードし、指定された秒数とサンプリングレートでリサンプル
def load_audio_file(file_path, length, sample_rate):
    data, sr = sf.read(file_path)
    # データが設定値よりも大きい場合は大きさを超えた分をカットする
    # データが設定値よりも小さい場合はデータの後ろを0でパディングする
    # シングルチャンネル(モノラル)の場合 (data.shape: [num_samples,])
    if data.ndim == 1:
        if len(data) > sample_rate*length:
            data = data[:sample_rate*length]
        else:
            data = np.pad(data, (0, max(0, sample_rate*length - len(data))), "constant")
        """data: (num_samples, )"""
    # マルチチャンネルの場合 (data.shape: [num_samples, num_channels])
    elif data.ndim == 2:
        if data.shape[0] > sample_rate*length:
            data = data[:sample_rate*length, :]
        else:
            data = np.pad(data, [(0, max(0, sample_rate*length-data.shape[0])), (0, 0)], "constant")
        """data: (num_samples, num_channels)"""
    else:
        print("number of audio channels are incorrect")
    return data


# 音声データを指定したサンプリングレートで保存
def save_audio_file(file_path, data, sampling_rate=16000):
    """"data: (num_samples, num_channels)"""
    sf.write(file_path, data, sampling_rate)


# 2つのオーディオデータを足し合わせる
def audio_mixer(data1, data2):
    assert len(data1) == len(data2)
    mixed_audio = data1 + data2
    return mixed_audio

# 音声データを振幅スペクトログラムと位相スペクトログラムに変換する
def wave_to_spec(data, fft_size, hop_length, win_length=None):
    # 短時間フーリエ変換(STFT)を行い、スペクトログラムを取得
    complex_spec = librosa.stft(data, n_fft=fft_size, hop_length=hop_length, win_length=win_length, window='hann')
    # amp_spec = np.abs(complex_spec) # 振幅スペクトログラムを取得
    # phase_spec = np.exp(1j * np.angle(complex_spec)) # 位相スペクトログラムを取得(フェーザ表示)
    # return amp_spec, phase_spec
    return complex_spec

# マルチチャンネルの音声データをスペクトログラムに変換する
def wave_to_spec_multi(data, sample_rate, fft_size, hop_length):
    """
    data: (num_channels, num_samples)
    sample_rate: sampling rate (int)
    fft_size: length of each segment (int)
    hop_length: shift size of each segment (int)
    """
    f, t, complex_spec = signal.stft(data, fs=sample_rate, window='hann', nperseg=fft_size, noverlap=fft_size-hop_length)
    """f: (freq_bins,), t: (time_frames,), spectrogram: (num_microphones, freq_bins, time_frames)"""
    amp_spec = np.abs(complex_spec) # 振幅スペクトログラムを取得
    # phase_spec = np.exp(1j * np.angle(complex_spec)) # 位相スペクトログラムを取得(フェーザ表示)
    return complex_spec, amp_spec

# スペクトログラムを音声データに変換する
def spec_to_wave(spec, hop_length):
    # 逆短時間フーリエ変換(iSTFT)を行い、スペクトログラムから音声データを取得
    wav_data = librosa.istft(spec, hop_length=hop_length)
    return wav_data

# waveファイルを読み込み波形のグラフを保存する
def wave_plot(input_path, output_path, fig_title=None):
    # open wave file
    wf = wave.open(input_path,'r')

    # load wave data
    length = 5 # 読み出すオーディオの長さ[s]
    rate = wf.getframerate()  # サンプリングレート[1/s]
    chunk_size = rate * length
    amp  = (2**8) ** wf.getsampwidth() / 2
    data = wf.readframes(chunk_size)   # バイナリ読み込み
    data = np.frombuffer(data,'int16') # intに変換
    data = data / amp                  # 振幅正規化

    # make time axis
    size = float(chunk_size)  # 波形サイズ
    x = np.arange(0, size/rate, 1.0/rate)

    # 図に描画
    sns.set() # スタイルをきれいにする
    fig = plt.figure(facecolor='w', linewidth=5, edgecolor='black')
    # ax = fig.add_subplot(1, 1, 1, ylim=(-0.5, 0.5)) # 図を1行目1列の1番目に表示(図を1つしか表示しない場合)
    ax = fig.add_subplot(1, 1, 1, title=fig_title) # 図を1行目1列の1番目に表示(図を1つしか表示しない場合)
    ax.set_xlabel('time[s]') # x軸名を設定
    ax.set_ylabel('magnitude') # y軸名を設定
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0)) # x軸の主目盛を1.0ごとに表示
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.10)) # y軸の主目盛を0.10ごとに表示
    file_name = os.path.basename(output_path).split('.')[0] # データの名前を設定
    ax.plot(x, data, label='{}'.format(file_name)) # データをプロット
    ax.legend(edgecolor="black") # 凡例を追加
    fig.savefig(output_path) # グラフを保存


if __name__ == "__main__":

    # audio_path = "./data/1-155858-E-25.wav"
    sample_rate = 16000
    audio_length = 3
    audio_channels = 8
    fft_size = 512
    hop_length = 160


    # spec_shape:[257, 301]
    # n_fft = 512 # 64ms
    # win_length = 400 # 25ms
    # hop_length = 160 # 10ms

    # spec_shape:[257, 376]
    # n_fft = 512
    # win_length = 512 # 32ms
    # hop_length = 128 # 8ms

    # spec_shape:[257, 301]
    # n_fft = 1024 # 64ms
    # win_length = None
    # hop_length = 256 # 16ms

    # audio_data = load_audio_file(audio_path, audio_length, audio_channels, sampling_rate)
    # spec_mag, spec_phase = wave_to_spec(audio_data, n_fft, hop_length, win_length)
    # print(spec_mag.shape)
    # print(spec_phase.shape)

    audio_path = "./test/p232_007/p232_007_mixed.wav"
    audio_data = load_audio_file(audio_path, audio_length, sample_rate)
    """audio_data: (num_samples=48000, num_channels=8)"""
    audio_data = audio_data.transpose(1, 0)
    """audio_data: (num_channels=8, num_samples=48000)"""
    multi_complex_spec = wave_to_spec_multi(audio_data, sample_rate, fft_size, hop_length)
    """multi_complex_spec: (num_channels=8, freq_bins=257, time_steps=301)"""
    print(multi_complex_spec.shape)
    
    # マルチチャンネル音声データをスペクトログラムに変換
    mulichannel_complex_spec = [] # それぞれのチャンネルの複素スペクトログラムを格納するリスト
    audio_data = load_audio_file(audio_path, audio_length, sample_rate)
    audio_data = np.asfortranarray(audio_data) # Fortran-contiguousに変換（これがないとエラーが出る）
    for i in range(audio_channels):
        # オーディオデータをスペクトログラムに変換
        complex_spec = wave_to_spec(audio_data[:, i], fft_size, hop_length)
        mulichannel_complex_spec.append(complex_spec)
    mulichannel_complex_spec = np.array(mulichannel_complex_spec)
    """mulichannel_complex_spec: (num_channels=8, freq_bins=257, time_steps=301)"""
    print(mulichannel_complex_spec.shape)


    # print(multi_complex_spec-mulichannel_complex_spec)


