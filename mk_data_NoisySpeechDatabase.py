import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import glob
import random

from tqdm import tqdm


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
def wave_to_spec(data, fft_size, hop_length):
    # 短時間フーリエ変換(STFT)を行い、スペクトログラムを取得
    spec = librosa.stft(data, n_fft=fft_size, hop_length=hop_length)
    mag = np.abs(spec) # 振幅スペクトログラムを取得
    phase = np.exp(1.j * np.angle(spec)) # 位相スペクトログラムを取得(フェーザ表示)
    # mel_spec = librosa.feature.melspectrogram(data, sr=sr, n_mels=128) # メルスペクトログラムを用いる場合はこっちを使う
    return mag, phase


if __name__ == '__main__':
    # 各パラメータを設定
    sampling_rate = 16000 # 作成するオーディオファイルのサンプリング周波数を指定
    audio_length = 3 # 単位は秒(second) → fft_size=1024,hop_length=768のとき、audio_length=6が最適化かも？
    audio_channels = 8
    voice_num_samples = 100 # 人の発話音声のファイル数
    env_noise_num_samples = 200 # 環境音のファイル数
    train_val_ratio = 0.9 # trainデータとvalidationデータの割合
    fft_size = 512 # 高速フーリエ変換のフレームサイズ
    hop_length = 160 # 高速フーリエ変換においてフレームをスライドさせる幅
    num_test_voice_samples = 10 # テスト用の人の発話音声のファイル数
    num_test_env_noise_samples = 10 # テスト用の環境音のファイル数
    noise_amplitude_decay = 0.3 # 環境音を混ぜる際の振幅の減衰率

    # 乱数を初期化
    random.seed(0)

    # マルチチャンネル音声をnpzファイルに格納する
    multichannel_audio_path = "./data/shokudo_noise/shokudo_split_3_sec_all"
    multichannel_audio_list = glob.glob(os.path.join(multichannel_audio_path, "*.wav"))
    multichannel_spec_path = "./data/shokudo_noise/shokudo_split_3_sec_all_spec"

    # 複数チャンネル分のスペクトログラムをまとめて一つのnpyファイルに保存する
    # npyファイルのshapeは[num_channels, freq_dim, time_frame]
    for multichannel_audio_path in tqdm(multichannel_audio_list):
        mulichannel_spec_mag = [] #　それぞれのチャンネルのスペクトログラム強度を格納するリスト
        audio_data = load_audio_file(multichannel_audio_path, audio_length, audio_channels, sampling_rate)
        audio_data = np.asfortranarray(audio_data) # これがないとエラーが出る
        for i in range(audio_channels):
            # オーディオデータをスペクトログラムに変換
            audio_mag, _ = wave_to_spec(audio_data[:, i], fft_size, hop_length)
            mulichannel_spec_mag.append(audio_mag)
        mulichannel_spec_mag = np.array(mulichannel_spec_mag) # shape:(8, 513, 188) [num_channels, freq_dim, time_frame]
        file_name = os.path.basename(multichannel_audio_path).split('.')[0]
        mulichannel_spec_save_path = os.path.join(multichannel_spec_path, file_name + ".npy")
        np.save(mulichannel_spec_save_path, mulichannel_spec_mag)

    # # データセットを格納するディレクトリを作成
    # save_dataset_dir = "./data/voice{}_noise{}/".format(voice_num_samples, env_noise_num_samples)
    # os.makedirs(save_dataset_dir, exist_ok=True)
    #
    # # 人の発話音声のファイルパスを無作為に取得
    # voice_data_path_template = "./data/jvs_ver1/jvs*/nonpara30/*/*.wav"
    # voice_list = random.sample(glob.glob(voice_data_path_template), voice_num_samples)
    # # データをtrainデータとvalidationデータに分割
    # voice_list_for_train = voice_list[:int(voice_num_samples*train_val_ratio)]
    # voice_list_for_val = voice_list[int(voice_num_samples*train_val_ratio):]
    # # test用のデータも別途作成
    # voice_list_for_test = random.sample(glob.glob(voice_data_path_template), num_test_voice_samples)
    #
    # # 環境音のファイルパスを無作為に取得
    # env_noise_path_template = "./data/environmental-sound-classification-50/audio/audio/16000/*.wav"
    # env_noise_list = random.sample(glob.glob(env_noise_path_template), env_noise_num_samples)
    # # データをtrainデータとvalidationデータに分割
    # env_noise_list_for_train = env_noise_list[:int(env_noise_num_samples*train_val_ratio)]
    # env_noise_list_for_val = env_noise_list[int(env_noise_num_samples*train_val_ratio):]
    # # test用のデータも別途作成
    # env_noise_list_for_test = random.sample(glob.glob(env_noise_path_template), num_test_env_noise_samples)
    # # trainデータ(学習用のスペクトログラム)を作成
    # print("trainデータ作成中")
    # train_data_path = os.path.join(save_dataset_dir, "train")
    # os.makedirs(train_data_path, exist_ok=True)
    # for voice_path in tqdm(voice_list_for_train):
    #     voice_file_name = voice_path.split('/')[-1] # (例)BASIC5000_0001.wav
    #     target_file_name = voice_file_name.split('.')[0] + "_target.npy" # (例)BASIC5000_0001_target.npy
    #     voice_data = load_audio_file(voice_path, audio_length, sampling_rate)
    #     # オーディオデータをスペクトログラムに変換
    #     voice_mag, _ = wave_to_spec(voice_data, fft_size, hop_length)
    #     # スペクトグラムを正規化
    #     max_mag = voice_mag.max()
    #     normed_voice_mag = voice_mag / max_mag
    #     # .npy形式でスペクトログラムを保存
    #     target_file_path = os.path.join(train_data_path, target_file_name)
    #     np.save(target_file_path, normed_voice_mag)
    #     # 人の発話音声それぞれに対して、複数の環境音を混ぜ合わせて混合音声を作成
    #     for idx, env_noise_path in enumerate(env_noise_list_for_train):
    #         noise_idx = str(idx).zfill(3)
    #         mixed_file_name = voice_file_name.split('.')[0] + "_" + noise_idx + "_mixed.npy" # (例)BASIC5000_0001_001_mixed.npy
    #         env_noise_data = load_audio_file(env_noise_path, audio_length, sampling_rate)
    #         env_noise_data = env_noise_data * noise_amplitude_decay # 環境音の大きさを小さくする
    #         mixed_audio_data = audio_mixer(voice_data, env_noise_data)
    #         # オーディオデータをスペクトログラムに変換
    #         mixed_mag, _ = wave_to_spec(mixed_audio_data, fft_size, hop_length)
    #         # スペクトグラムを正規化(雑音を混ぜる前と混ぜた後で人の声の音量を一致させるためmax_specで割る)
    #         normed_mixed_mag = mixed_mag / max_mag
    #         # .npy形式でスペクトログラムを保存
    #         mixed_file_path = os.path.join(train_data_path, mixed_file_name)
    #         np.save(mixed_file_path, normed_mixed_mag)
    #
    # # validationデータ(評価用のスペクトログラム)を作成
    # print("validationデータ作成中")
    # val_data_path = os.path.join(save_dataset_dir, "val")
    # os.makedirs(val_data_path, exist_ok=True)
    # for voice_path in tqdm(voice_list_for_val):
    #     voice_file_name = voice_path.split('/')[-1] # (例)BASIC5000_0001.wav
    #     target_file_name = voice_file_name.split('.')[0] + "_target.npy" # (例)BASIC5000_0001_target.npy
    #     voice_data = load_audio_file(voice_path, audio_length, sampling_rate)
    #     # オーディオデータをスペクトログラムに変換
    #     voice_mag, _ = wave_to_spec(voice_data, fft_size, hop_length)
    #     # スペクトログラムを正規化
    #     max_mag = voice_mag.max()
    #     normed_voice_mag = voice_mag / max_mag
    #     # .npy形式でスペクトログラムを保存
    #     target_file_path = os.path.join(val_data_path, target_file_name)
    #     np.save(target_file_path, normed_voice_mag)
    #     # 人の発話音声それぞれに対して、複数の環境音を混ぜ合わせて混合音声を作成
    #     for idx, env_noise_path in enumerate(env_noise_list_for_val):
    #         noise_idx = str(idx).zfill(3)
    #         mixed_file_name = voice_file_name.split('.')[0] + "_" + noise_idx + "_mixed.npy" # (例)BASIC5000_0001_001_mixed.npy
    #         env_noise_data = load_audio_file(env_noise_path, audio_length, sampling_rate)
    #         env_noise_data = env_noise_data * noise_amplitude_decay # 環境音の大きさを小さくする
    #         mixed_audio_data = audio_mixer(voice_data, env_noise_data)
    #         # オーディオデータをスペクトログラムに変換
    #         mixed_mag, _ = wave_to_spec(mixed_audio_data, fft_size, hop_length)
    #         # スペクトグラムを正規化(雑音を混ぜる前と混ぜた後で人の声の音量を一致させるためmax_specで割る)
    #         normed_mixed_mag = mixed_mag / max_mag
    #         # .npy形式でスペクトログラムを保存
    #         mixed_file_path = os.path.join(val_data_path, mixed_file_name)
    #         np.save(mixed_file_path, normed_mixed_mag)
    #
    # # testデータ(テスト用のオーディオファイル)を作成
    # print("testデータ作成中")
    # test_data_path = os.path.join(save_dataset_dir, "test")
    # os.makedirs(test_data_path, exist_ok=True)
    # for voice_path in tqdm(voice_list_for_test):
    #     voice_file_name = voice_path.split('/')[-1] # BASIC5000_0001.wav
    #     target_file_name = voice_file_name.split('.')[0] + "_target.wav" # BASIC5000_0001_target.wav
    #     voice_data = load_audio_file(voice_path, audio_length, sampling_rate)
    #     # 元の音声データのサンプリング周波数を指定して保存
    #     target_file_path = os.path.join(test_data_path, target_file_name)
    #     save_audio_file(target_file_path, voice_data, sampling_rate)
    #     # 人の発話音声それぞれに対して、複数の環境音を混ぜ合わせて混合音声を作成
    #     for idx, env_noise_path in enumerate(env_noise_list_for_test):
    #         noise_idx = str(idx).zfill(3)
    #         mixed_file_name = voice_file_name.split('.')[0] + "_" + noise_idx + "_mixed.wav"
    #         env_noise_data = load_audio_file(env_noise_path, audio_length, sampling_rate)
    #         env_noise_data = env_noise_data * noise_amplitude_decay # 環境音の大きさを小さくする
    #         mixed_audio_data = audio_mixer(voice_data, env_noise_data)
    #         # 混合した音声データのサンプリング周波数を指定して保存
    #         mixed_file_path = os.path.join(test_data_path, mixed_file_name)
    #         save_audio_file(mixed_file_path, mixed_audio_data, sampling_rate)
    #         # 元の雑音データも保存
    #         noise_file_name = voice_file_name.split('.')[0] + "_" + noise_idx + "_noise.wav"
    #         noise_file_path = os.path.join(test_data_path, noise_file_name)
    #         save_audio_file(noise_file_path, env_noise_data, sampling_rate)
    #
    # print("データ作成完了　保存先：{}".format(save_dataset_dir))
