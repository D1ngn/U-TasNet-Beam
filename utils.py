import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

#　音声データをロードし、指定された秒数とサンプリングレートでリサンプル
def load_audio_file(file_path, length, num_channels, sampling_rate=16000):
    data, sr = sf.read(file_path)
    # データが設定値よりも大きい場合は大きさを超えた分をカットする
    # データが設定値よりも小さい場合はデータの後ろを0でパディングする
    # 1ch(モノラル)の場合
    if num_channels == 1:
        if len(data) > sampling_rate*length:
            data = data[:sampling_rate*length]
        else:
            data = np.pad(data, (0, max(0, sampling_rate*length - len(data))), "constant")
    # マルチチャンネルの場合
    elif num_channels > 1:
        if data.shape[0] > sampling_rate*length:
            data = data[:sampling_rate*length, :]
        else:
            data = np.pad(data, [(0, max(0, sampling_rate*length-data.shape[0])), (0, 0)], "constant")
    else:
        print("please designate correct num_channels")
    return data


# 音声データを指定したサンプリングレートで保存
def save_audio_file(file_path, data, sampling_rate=16000):
    # librosa.output.write_wav(file_path, data, sampling_rate) # 正常に動作しないので変更
    sf.write(file_path, data, sampling_rate)

# 2つのオーディオデータを足し合わせる
def audio_mixer(data1, data2):
    assert len(data1) == len(data2)
    mixed_audio = data1 + data2
    return mixed_audio

# 音声データをスペクトログラムに変換する
def wave_to_spec(data, fft_size, hop_length, win_length=None):
    # 短時間フーリエ変換(STFT)を行い、スペクトログラムを取得
    spec = librosa.stft(data, n_fft=fft_size, hop_length=hop_length, win_length=win_length)
    mag = np.abs(spec) # 振幅スペクトログラムを取得
    phase = np.exp(1.j * np.angle(spec)) # 位相スペクトログラムを取得(フェーザ表示)
    # mel_spec = librosa.feature.melspectrogram(data, sr=sr, n_mels=128) # メルスペクトログラムを用いる場合はこっちを使う
    return mag, phase

# スペクトログラムを音声データに変換する
def spec_to_wav(spec, hop_length):
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

    audio_path = "./data/1-155858-E-25.wav"
    sampling_rate = 16000
    audio_length = 3
    audio_channels = 1

    # spec_shape:[257, 301]
    # n_fft = 512 # 64ms
    # win_length = 400 # 25ms
    # hop_length = 160 # 10ms

    # spec_shape:[257, 376]
    # n_fft = 512
    # win_length = 512 # 32ms
    # hop_length = 128 # 8ms

    # spec_shape:[257, 301]
    n_fft = 1024 # 64ms
    win_length = None
    hop_length = 256 # 16ms

    audio_data = load_audio_file(audio_path, audio_length, audio_channels, sampling_rate)
    spec_mag, spec_phase = wave_to_spec(audio_data, n_fft, hop_length, win_length)
    print(spec_mag.shape)
    print(spec_phase.shape)
