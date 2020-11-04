#!/usr/bin/env python
# coding: utf-8

# 必要モジュールのimport
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import wave
import subprocess
import time

from models import MCDUnet_kernel3

from natsort import natsorted
from tqdm import tqdm

# 音声評価用
import sys
sys.path.append('..')
from MyLibrary.MyFunc import wave_plot, audio_eval, load_audio_file, save_audio_file, wave_to_spec

# モデルのパラメータ数をカウント
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# スペクトログラムを音声データに変換する
def spec_to_wav(spec, hop_length):
    # 逆短時間フーリエ変換(iSTFT)を行い、スペクトログラムから音声データを取得
    wav_data = librosa.istft(spec, hop_length=hop_length)
    return wav_data


# スペクトログラムを図にプロットする関数 今は使っていない
def spec_plot_old(input_spec, save_path):
    # パワースペクトルを対数パワースペクトルに変換
    log_power_spec = librosa.amplitude_to_db(input_spec, ref=np.max)
    plt.figure(figsize=(12,5)) # 図の大きさを指定
    librosa.display.specshow(log_power_spec, x_axis='time', y_axis='log')
    plt.title('Spectroram') # タイトル
    plt.xlabel("time[s]") # 横軸のラベル
    plt.ylabel("amplitude") # 縦軸のラベル
    plt.colorbar(format='%+02.0f dB') # カラーバー表示
    plt.savefig(save_path)

# スペクトログラムを図にプロットする関数
def spec_plot(base_dir, wav_path, save_path, audio_length):
    # soxコマンドによりwavファイルからスペクトログラムの画像を生成
    cmd1 = "sox {} -n trim 0 {} rate 16.0k spectrogram".format(wav_path, audio_length)
    subprocess.call(cmd1, shell=True)
    # 生成されたスペクトログラム画像を移動
    #(inference.pyを実行したディレクトリにスペクトログラムが生成されてしまうため)
    spec_path = os.path.join(base_dir, "spectrogram.png")
    cmd2 = "mv {} {}".format(spec_path, save_path)
    subprocess.call(cmd2, shell=True)

if __name__ == '__main__':

    sampling_rate = 16000 # 作成するオーディオファイルのサンプリング周波数を指定
    audio_length = 3 # 単位は秒(second) → fft_size=1024,hop_length=768のとき、audio_length=6が最適かも？
    fft_size = 512 # 高速フーリエ変換のフレームサイズ
    hop_length = 160 # 高速フーリエ変換におけるフレームのスライド幅
    spec_frame_num = 64 # スペクトログラムのフレーム数 spec_freq_dim=512のとき、音声の長さが5秒の場合は128, 3秒の場合は64

    # 音声ファイルのディレクトリを指定
#     target_voice_dir = "../data/NoisySpeechDatabase/clean_testset_wav_16kHz/"
#     interference_audio_dir = "../data/NoisySpeechDatabase/interference_testset_wav_16kHz/"
#     mixed_audio_dir = "../data/NoisySpeechDatabase/noisy_testset_wav_16kHz/"
    test_data_dir = "../data/NoisySpeechDataset_for_unet_fft_512_8ch_1007/test"

#     target_voice_path_list = natsorted(glob.glob(os.path.join(test_data_dir, "*_target.wav")))
#     interference_audio_path_list = natsorted(glob.glob(os.path.join(test_data_dir, "*_interference.wav")))
    mixed_audio_path_list = natsorted(glob.glob(os.path.join(test_data_dir, "*_mixed.wav")))

    # 学習済みのパラメータを保存したチェックポイントファイルのパスを指定
    checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_fft_512_kernel3_multi_1007/ckpt_epoch200.pt"
    # ネットワークモデルを指定
    model = MCDUnet_kernel3()
    # GPUが使える場合はGPUを使用、使えない場合はCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：" , device)
    # 学習済みのパラメータをロード
    model_params = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_params['model_state_dict'])
    # print("モデルのパラメータ数：", count_parameters(model))
    # Unetを使って推論
    # ネットワークを推論モードへ
    model.eval()

    # 音声評価結果の合計値を格納するリストを用意
    sdr_mix_list = []
    sir_mix_list = []
    sar_mix_list = []
    sdr_est_list = []
    sir_est_list = []
    sar_est_list = []

    # 分離処理の開始時間
    start_time = time.perf_counter()
    for mixed_audio_path in tqdm(mixed_audio_path_list):
        # 音声データをロード(現在は学習時と同じ処理をしているが、いずれはマイクロホンのリアルストリーミング音声を入力にしたい)
        mixed_audio_data = load_audio_file(mixed_audio_path, audio_length, sampling_rate)
        mixed_audio_data = np.asfortranarray(mixed_audio_data) # これがないとエラーが出る
        """mixed_audio_data: (num_samples, num_channels=8)"""
        # マルチチャンネルオーディオデータをスペクトログラムに変換
        multichannel_mixed_amp_spec = [] # 振幅スペクトログラム
        multichannel_mixed_phase_spec = [] # 位相スペクトログラム
        for i in range(mixed_audio_data.shape[1]):
            # オーディオデータをスペクトログラムに変換
            mixed_amp, mixed_phase = wave_to_spec(mixed_audio_data[:, i], fft_size, hop_length)
            multichannel_mixed_amp_spec.append(mixed_amp)
            multichannel_mixed_phase_spec.append(mixed_phase)
        multichannel_mixed_amp_spec = np.array(multichannel_mixed_amp_spec)
        """multichannel_mixed_amp_spec: (num_channels=8, freq_bins=257, time_frames=301)"""
        multichannel_mixed_phase_spec = np.array(multichannel_mixed_phase_spec)
        """multichannel_mixed_phase_spec: (num_channels=8, freq_bins=257, time_frames=301)"""
        # 振幅スペクトログラムを正規化
        max_amp = multichannel_mixed_amp_spec.max()
        normed_multichannel_mixed_amp_spec = multichannel_mixed_amp_spec / max_amp
        # データの形式をモデルに入力できる形式に変更する
        # モデルの入力サイズに合わせてpadding
        amp_spec_padded = np.pad(normed_multichannel_mixed_amp_spec, [(0, 0), (0, 0), (0, 513-normed_multichannel_mixed_amp_spec.shape[2])], 'constant')
        phase_spec_padded = np.pad(multichannel_mixed_phase_spec, [(0, 0), (0, 0), (0, 513-multichannel_mixed_phase_spec.shape[2])], 'constant')
        # 1次元目に次元を追加
        amp_spec_expanded = amp_spec_padded[np.newaxis, :, :, :]
        """amp_spec_expanded: (1, 8, 257, 513)"""
        phase_spec_expanded = phase_spec_padded[np.newaxis, :, :, :]
        """phase_spec_expanded: (1, 8, 257, 513)"""
        # numpy形式のデータをpytorchのテンソルに変換
        amp_spec_tensor = torch.from_numpy(amp_spec_expanded.astype(np.float32)).clone()
        # 環境音のmaskを計算
        mask = model(amp_spec_tensor)
        # pytorchのtensorをnumpy配列に変換
#         mask = mask.to(device).detach().numpy().copy() # GPU
        mask = mask.detach().numpy().copy() # CPU
        # 人の声を取り出す
        normed_estimated_amp_spec = mask * amp_spec_expanded
        # 正規化によって小さくなった音量を元に戻す
        estimated_amp_spec = normed_estimated_amp_spec * max_amp
        # マスクした後の振幅スペクトログラムに入力音声の位相スペクトログラムを掛け合わせて音声を復元
        voice_spec = estimated_amp_spec * phase_spec_expanded
        """voice_spec: (batch_size=1, num_channels=8, freq_bins=257, time_frames=513)"""
        voice_spec = np.squeeze(voice_spec)
        """voice_spec: (num_channels=8, freq_bins=257, time_frames=513)"""
        # 前処理の際にpaddingしたスペクトログラムを元の形に戻す
        voice_spec = voice_spec[:, :, :normed_multichannel_mixed_amp_spec.shape[2]]
        """voice_spec: (num_channels=8, freq_bins=257, time_frames=301)"""
        # マルチチャンネルスペクトログラムを音声波形に変換
        multichannel_estimated_voice_data= np.zeros(mixed_audio_data.shape, dtype='float32') # マルチチャンネル音声波形を格納する配列
        for i in range(voice_spec.shape[0]):
            # 1chごとスペクトログラムを音声波形に変換
            estimated_voice_data = spec_to_wav(voice_spec[i, :, :], hop_length)
            multichannel_estimated_voice_data[:, i] = estimated_voice_data
        """multichannel_estimated_voice_data: (num_samples, num_channels=8)"""
        # オーディオデータを保存
        estimated_voice_path = "./estimated_voice.wav"
        save_audio_file(estimated_voice_path, multichannel_estimated_voice_data, sampling_rate=16000)
        file_num = os.path.basename(mixed_audio_path).split('.')[0].rsplit('_', maxsplit=1)[0] # p257_013
        target_voice_path = os.path.join(test_data_dir, file_num + "_target.wav")
        interference_audio_path = os.path.join(test_data_dir, file_num + "_interference.wav")

        # 音声評価
        sdr_mix, sir_mix, sar_mix, sdr_est, sir_est, sar_est = \
        audio_eval(audio_length, target_voice_path, interference_audio_path, mixed_audio_path, estimated_voice_path)
        # 音声評価結果を記録
        sdr_mix_list.append(sdr_mix)
        sir_mix_list.append(sir_mix)
        sar_mix_list.append(sar_mix)
        sdr_est_list.append(sdr_est)
        sir_est_list.append(sir_est)
        sar_est_list.append(sar_est)
        # 推定音声が蓄積されないように削除
        os.remove(estimated_voice_path)

    # データの数を取得
    num_file = len(mixed_audio_path_list)
    # 分離処理の終了時間
    finish_time = time.perf_counter()
    # 処理時間
    process_time = finish_time - start_time
    print("合計処理時間：", str(process_time) + 'sec')
    print("平均処理時間：", str(process_time/num_file) + 'sec')
    print("平均 | SDR_mix: {:.3f}, SIR_mix: {:.3f}, SAR_mix: {:.3f}".format(np.mean(sdr_mix_list), np.mean(sir_mix_list), np.mean(sar_mix_list)))
    print("平均 | SDR_est: {:.3f}, SIR_est: {:.3f}, SAR_est: {:.3f}".format(np.mean(sdr_est_list), np.mean(sir_est_list), np.mean(sar_est_list)))
    print("標準偏差 | SDR_mix: {:.3f}, SIR_mix: {:.3f}, SAR_mix: {:.3f}".format(np.std(sdr_mix_list), np.std(sir_mix_list), np.std(sar_mix_list)))
    print("標準偏差 | SDR_est: {:.3f}, SIR_est: {:.3f}, SAR_est: {:.3f}".format(np.std(sdr_est_list), np.std(sir_est_list), np.std(sar_est_list)))
