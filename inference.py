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

from models import FCMaskEstimator, BLSTMMaskEstimator, UnetMaskEstimator_kernel3
from beamformer import estimate_covariance_matrix, condition_covariance, estimate_steering_vector, mvdr_beamformer, gev_beamformer, sparse, ds_beamformer, mwf

from natsort import natsorted
from tqdm import tqdm

# 音声処理用
import sys
sys.path.append('..')
from MyLibrary.MyFunc import wave_plot, audio_eval, load_audio_file, save_audio_file, wave_to_spec, spec_to_wave
from training import AudioProcess

# モデルのパラメータ数をカウント
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

# データを標準化（平均0、分散1に正規化（Z-score Normalization））
def standardize(data):
    data_mean = data.mean(keepdims=True)
    data_std = data.std(keepdims=True, ddof=0) # 母集団の標準偏差（標本標準偏差を使用する場合はddof=1）
    standardized_data = (data - data_mean) / data_std
    return standardized_data

if __name__ == '__main__':
    # 各パラメータを設定
    sample_rate = 16000 # 作成するオーディオファイルのサンプリング周波数を指定
    audio_length = 3 # 単位は秒(second) → fft_size=1024,hop_length=768のとき、audio_length=6が最適かも？
    fft_size = 512 # 高速フーリエ変換のフレームサイズ
    hop_length = 160 # 高速フーリエ変換におけるフレームのスライド幅
    spec_frame_num = 64 # スペクトログラムのフレーム数 spec_freq_dim=512のとき、音声の長さが5秒の場合は128, 3秒の場合は64
    # マスクのチャンネルを指定（いずれはconfigまたはargsで指定）TODO
    target_aware_channel = 0
    noise_aware_channel = 4
    
    # 英語（男性）
    # target_voice_file = "./test/p232_016/p232_016_target.wav"
    # interference_audio_file = "./test/p232_016/p232_016_interference_azimuth45.wav"
    # mixed_audio_file = "./test/p232_016/p232_016_mixed_azimuth45.wav"
    # target_voice_file = "./test/p232_021/p232_021_target.wav"
    # interference_audio_file = "./test/p232_021/p232_021_interference_azimuth15.wav"
    # mixed_audio_file = "./test/p232_021/p232_021_mixed_azimuth15.wav"
    # 英語（女性）
    target_voice_file = "./test/p257_006/p257_006_target.wav"
    interference_audio_file = "./test/p257_006/p257_006_interference_azimuth60.wav"
    mixed_audio_file = "./test/p257_006/p257_006_mixed_azimuth60.wav"
    # target_voice_file = "./test/p257_130/p257_130_target.wav"
    # interference_audio_file = "./test/p257_130/p257_130_interference_azimuth0.wav"
    # mixed_audio_file = "./test/p257_130/p257_130_mixed_azimuth0.wav"
    # 日本語（女性）
    # target_voice_file = "./test/JVS/BASIC5000_0145_target.wav"
    # interference_audio_file = "./test/JVS/BASIC5000_0145_interference.wav"
    # mixed_audio_file = "./test/JVS/BASIC5000_0145_mixed.wav"

    wave_dir = "./output/wave/"
    os.makedirs(wave_dir, exist_ok=True)
    # オーディオファイルに対応する音声の波形を保存
    wave_image_dir = "./output/wave_image/"
    os.makedirs(wave_image_dir, exist_ok=True)
    # オーディオファイルに対応するスペクトログラムを保存
    spec_dir = "./output/spectrogram/"
    os.makedirs(spec_dir, exist_ok=True)

    # 学習済みのパラメータを保存したチェックポイントファイルのパスを指定
    # NoisySpeechDataset_for_unet_fft_512_multi_wav_1207で学習
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_all_BLSTM_1202/ckpt_epoch80.pt" # BLSTM
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_all_FC_1202/ckpt_epoch80.pt" # FC
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_BLSTM_1201/ckpt_epoch70.pt" # BLSTM small
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_BLSTM_1211/ckpt_epoch80.pt" # BLSTM small2
    checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_Unet_aware_1208/ckpt_epoch110.pt" # U-Net aware channel←ベストモデル
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_Unet_aware_1211/ckpt_epoch160.pt" # U-Net aware channel2
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_Unet_median_1209/ckpt_epoch50.pt" # U-Net median operation
    # NoisySpeechDataset_for_unet_fft_512_multi_wav_1209で学習
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_logmel_BLSTM_1209/ckpt_epoch100.pt" # BLSTM-logmel
    # マスク推定モデルのタイプを指定
    model_type = 'Unet' # 'FC' or 'BLSTM' or 'Unet'
    # ビームフォーマのタイプを指定
    beamformer_type = 'MVDR' # 'DS' or 'MVDR' or 'GEV', or 'MWF' or 'Sparse'

    # ネットワークモデルを定義
    if model_type == 'BLSTM':
        model = BLSTMMaskEstimator()
    elif model_type == 'FC':
        model = FCMaskEstimator()
    elif model_type == 'Unet':
        model = UnetMaskEstimator_kernel3()
        pass
    # 前処理クラスのインスタンスを作成
    transform = AudioProcess(sample_rate, fft_size, hop_length, model_type)
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

    # 処理の開始時間
    start_time = time.perf_counter()
    # 音声データをロード
    mixed_audio_data = load_audio_file(mixed_audio_file, audio_length, sample_rate)
    """mixed_audio_data: (num_samples, num_channels)"""
    # マルチチャンネル音声データを複素スペクトログラムと振幅スペクトログラムに変換
    mixed_complex_spec, mixed_amp_spec = transform(mixed_audio_data)
    """mixed_complex_spec: (num_channels, freq_bins, time_steps), mixed_amp_spec: (num_channels, freq_bins, time_steps)"""
    # 振幅スペクトログラムを標準化
    mixed_amp_spec = standardize(mixed_amp_spec)
    # numpy形式のデータをpytorchのテンソルに変換
    mixed_amp_spec = torch.from_numpy(mixed_amp_spec.astype(np.float32)).clone()
    # モデルに入力できるようにバッチサイズの次元を追加
    mixed_amp_spec = mixed_amp_spec.unsqueeze(0)
    """mixed_amp_spec: (batch_size, num_channels, freq_bins, time_steps)"""
    # 音源方向推定情報を含むマスクを推定
    target_mask_output, noise_mask_output = model(mixed_amp_spec)
    if model_type == 'FC' or 'Unet':
        # マスクのチャンネルを指定（目的音に近いチャンネルと雑音に近いチャンネル）
        estimated_target_mask = target_mask_output[:, target_aware_channel, :, :]
        """estimated_target_mask: (batch_size, freq_bins, time_steps)"""
        estimated_noise_mask = noise_mask_output[:, noise_aware_channel, :, :]
        """estimated_noise_mask: (batch_size, freq_bins, time_steps)"""
    elif model_type == 'BLSTM':
        # 複数チャンネル間のマスク値の中央値をとる（median pooling）
        (estimated_target_mask, _) = torch.median(target_mask_output, dim=1)
        """estimated_target_mask: (batch_size, freq_bins, time_steps)"""
        (estimated_noise_mask, _) = torch.median(noise_mask_output, dim=1)
        """estimated_noise_mask: (batch_size, freq_bins, time_steps)"""
    else:
        print("Please specify the correct model type")
    # バッチサイズの次元を削除
    estimated_target_mask = estimated_target_mask.squeeze(0)
    """estimated_target_mask: (freq_bins, time_steps)"""
    estimated_noise_mask = estimated_noise_mask.squeeze(0)
    """estimated_noise_mask: (freq_bins, time_steps)"""
    # pytorchのテンソルをnumpy形式のデータに変換
    estimated_target_mask = estimated_target_mask.detach().numpy().copy() # CPU
    estimated_noise_mask = estimated_noise_mask.detach().numpy().copy() # CPU
    
    # U-Netの場合paddingされた分を削除する
    if model_type == 'Unet':
        # とりあえずハードコーディング TODO
        mixed_complex_spec = mixed_complex_spec[:, :, :301]
        estimated_target_mask = estimated_target_mask[:, :301] 
        estimated_noise_mask = estimated_noise_mask[:, :301]
    # 目的音のマスクと雑音のマスクからそれぞれの空間共分散行列を推定
    target_covariance_matrix = estimate_covariance_matrix(mixed_complex_spec, estimated_target_mask)
    noise_covariance_matrix = estimate_covariance_matrix(mixed_complex_spec, estimated_noise_mask)
    noise_covariance_matrix = condition_covariance(noise_covariance_matrix, 1e-6) # これがないと性能が大きく落ちる（雑音の共分散行列のみで良い）
    # noise_covariance_matrix /= np.trace(noise_covariance_matrix, axis1=-2, axis2=-1)[..., None, None]
    # ビームフォーマによる雑音除去を実行
    if beamformer_type == 'MVDR':
        # target_steering_vectors = estimate_steering_vector(target_covariance_matrix)
        # estimated_spec = mvdr_beamformer(mixed_complex_spec, target_steering_vectors, noise_covariance_matrix)
        estimated_spec = mvdr_beamformer(mixed_complex_spec, target_covariance_matrix, noise_covariance_matrix)
    elif beamformer_type == 'GEV':
        estimated_spec = gev_beamformer(mixed_complex_spec, target_covariance_matrix, noise_covariance_matrix)
    elif beamformer_type == "DS":
        target_steering_vectors = estimate_steering_vector(target_covariance_matrix)
        estimated_spec = ds_beamformer(mixed_complex_spec, target_steering_vectors)
    elif beamformer_type == "MWF":
        estimated_spec = mwf(mixed_complex_spec, target_covariance_matrix, noise_covariance_matrix)
    elif beamformer_type == 'Sparse':
        estimated_spec = sparse(mixed_complex_spec, estimated_target_mask) # マスクが正常に推定できているかどうかをテストする用
    else:
        print("Please specify the correct beamformer type")
    """estimated_spec: (num_channels, freq_bins, time_frames)"""

    # マルチチャンネルスペクトログラムを音声波形に変換
    multichannel_estimated_voice_data= np.zeros(mixed_audio_data.shape, dtype='float64') # マルチチャンネル音声波形を格納する配列
    # 1chごとスペクトログラムを音声波形に変換
    for i in range(estimated_spec.shape[0]):
        # estimated_voice_data = spec_to_wave(estimated_spec[i, :, :], hop_length)
        estimated_voice_data = librosa.core.istft(estimated_spec[i, :, :], hop_length=hop_length)
        multichannel_estimated_voice_data[:, i] = estimated_voice_data
    """multichannel_estimated_voice_data: (num_samples, num_channels)"""
    # 処理の終了時間
    finish_time = time.perf_counter()
    # 処理時間
    process_time = finish_time - start_time
    print("処理時間：", str(process_time) + 'sec')

    # オーディオデータを保存
    estimated_voice_path = os.path.join(wave_dir, "estimated_voice.wav")
    save_audio_file(estimated_voice_path, multichannel_estimated_voice_data, sample_rate=16000)
    # デバッグ用に元のオーディオデータとそのスペクトログラムを保存
    # オリジナル音声
    target_voice_path = os.path.join(wave_dir, "target_voice.wav")
    target_voice_data = load_audio_file(target_voice_file, audio_length, sample_rate)
    save_audio_file(target_voice_path, target_voice_data, sample_rate=16000)
    # 外的雑音
    interference_audio_path = os.path.join(wave_dir, "interference_audio.wav")
    interference_audio_data = load_audio_file(interference_audio_file, audio_length, sample_rate)
    save_audio_file(interference_audio_path, interference_audio_data, sample_rate=16000)
    # 混合音声
    mixed_audio_path = os.path.join(wave_dir, "mixed_audio.wav")
    save_audio_file(mixed_audio_path, mixed_audio_data, sample_rate=16000)

    # # 音声の波形を画像として保存（マルチチャンネル未対応）
    # # オリジナル音声の波形
    # target_voice_img_path = os.path.join(wave_image_dir, "target_voice.png")
    # wave_plot(target_voice_path, target_voice_img_path, audio_length, ylim_min=-1.0, ylim_max=1.0)
    # # 外的雑音の波形
    # interference_img_path = os.path.join(wave_image_dir, "interference_audio.png")
    # wave_plot(interference_audio_path, interference_img_path, audio_length, ylim_min=-1.0, ylim_max=1.0)
    # # 分離音の波形
    # estimated_voice_img_path = os.path.join(wave_image_dir, "estimated_voice.png")
    # wave_plot(estimated_voice_path, estimated_voice_img_path, audio_length, ylim_min=-1.0, ylim_max=1.0)
    # # 混合音声の波形
    # mixed_audio_img_path = os.path.join(wave_image_dir, "mixed_audio.png")
    # wave_plot(mixed_audio_path, mixed_audio_img_path, audio_length, ylim_min=-1.0, ylim_max=1.0)

    # スペクトログラムを画像として保存
    # 現在のディレクトリ位置を取得
    base_dir = os.getcwd()
    # オリジナル音声のスペクトログラム
    target_voice_spec_path = os.path.join(spec_dir, "target_voice.png")
    spec_plot(base_dir, target_voice_path, target_voice_spec_path, audio_length)
    # 外的雑音のスペクトログラム
    interference_audio_spec_path = os.path.join(spec_dir, "interference_audio.png")
    spec_plot(base_dir, interference_audio_path, interference_audio_spec_path, audio_length)
    # 分離音のスペクトログラム
    estimated_voice_spec_path = os.path.join(spec_dir, "estimated_voice.png")
    spec_plot(base_dir, estimated_voice_path, estimated_voice_spec_path, audio_length)
    # 混合音声のスペクトログラム
    mixed_audio_spec_path = os.path.join(spec_dir, "mixed_audio.png")
    spec_plot(base_dir, mixed_audio_path, mixed_audio_spec_path, audio_length)

    # 音声評価
    sdr_mix, sir_mix, sar_mix, sdr_est, sir_est, sar_est = \
        audio_eval(audio_length, sample_rate, target_voice_path, interference_audio_path, mixed_audio_path, estimated_voice_path)
    print("SDR_mix: {:.3f}, SIR_mix: {:.3f}, SAR_mix: {:.3f}".format(sdr_mix, sir_mix, sar_mix))
    print("SDR_est: {:.3f}, SIR_est: {:.3f}, SAR_est: {:.3f}".format(sdr_est, sir_est, sar_est))
