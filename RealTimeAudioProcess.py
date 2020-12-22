# -*- coding:utf-8 -*-

# 必要モジュールのimport
# pytorch関連
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import os
import sys
import numpy as np
import pyaudio
import wave
import struct
import librosa
import argparse
import time
import queue
import sounddevice as sd
import soundfile as sf

# マスクビームフォーマ関連
from models import FCMaskEstimator, BLSTMMaskEstimator, UnetMaskEstimator_kernel3
from beamformer import estimate_covariance_matrix, condition_covariance, estimate_steering_vector, mvdr_beamformer, gev_beamformer, sparse, ds_beamformer, mwf
from training import AudioProcess

from utils import wave_to_spec, spec_to_wave, save_audio_file

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

# データを標準化（平均0、分散1に正規化（Z-score Normalization））
def standardize(data):
    data_mean = data.mean(keepdims=True)
    data_std = data.std(keepdims=True, ddof=0) # 母集団の標準偏差（標本標準偏差を使用する場合はddof=1）
    standardized_data = (data - data_mean) / data_std
    return standardized_data

# 混合音の振幅スペクトログラムから目的音と雑音のマスクを推定
def estimate_mask(mixed_amp_spec):
    # 振幅スペクトログラムを標準化
    mixed_amp_spec = standardize(mixed_amp_spec)
    # numpy形式のデータをpytorchのテンソルに変換
    mixed_amp_spec = torch.from_numpy(mixed_amp_spec.astype(np.float32)).clone()
    # モデルに入力できるようにバッチサイズの次元を追加
    mixed_amp_spec = mixed_amp_spec.unsqueeze(0)
    """mixed_amp_spec: (batch_size, num_channels, freq_bins, time_steps)"""
    # 音源方向推定情報を含むマスクを推定
    target_mask, noise_mask = model(mixed_amp_spec)
    if args.model_type == 'FC' or 'Unet':
        # マスクのチャンネルを指定（目的音に近いチャンネルと雑音に近いチャンネル）
        estimated_target_mask = target_mask[:, args.target_aware_channel, :, :]
        """estimated_target_mask: (batch_size, freq_bins, time_steps)"""
        estimated_noise_mask = noise_mask[:, args.noise_aware_channel, :, :]
        """estimated_noise_mask: (batch_size, freq_bins, time_steps)"""
    elif args.model_type == 'BLSTM':
        # 複数チャンネル間のマスク値の中央値をとる（median pooling）
        (estimated_target_mask, _) = torch.median(target_mask, dim=1)
        """estimated_target_mask: (batch_size, freq_bins, time_steps)"""
        (estimated_noise_mask, _) = torch.median(noise_mask, dim=1)
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
    return estimated_target_mask, estimated_noise_mask

# マイクロホンで取得した音声を固定時間に分割し、処理を行う
def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # マイクロホンのゲイン調整
    indata = indata * args.mic_gain
    """indata: (num_samples, num_channels)"""
    # 雑音除去を行う場合
    if args.denoising_mode:
        # Fancy indexing with mapping creates a (necessary!) copy
        # マルチチャンネル音声データを複素スペクトログラムと振幅スペクトログラムに変換
        mixed_complex_spec, mixed_amp_spec = transform(indata.copy())
        """mixed_complex_spec: (num_channels, freq_bins, time_steps), mixed_amp_spec: (num_channels, freq_bins, time_steps)"""
        # 目的音と雑音のマスクを推定
        estimated_target_mask, estimated_noise_mask = estimate_mask(mixed_amp_spec)
        # U-Netの場合paddingされた分を削除する
        if args.model_type == 'Unet':
            # とりあえずハードコーディング TODO
            mixed_complex_spec = mixed_complex_spec[:, :, :301]
            estimated_target_mask = estimated_target_mask[:, :301] 
            estimated_noise_mask = estimated_noise_mask[:, :301]
        # 目的音のマスクと雑音のマスクから空間相関行列を推定
        target_covariance_matrix = estimate_covariance_matrix(mixed_complex_spec, estimated_target_mask)
        noise_covariance_matrix = estimate_covariance_matrix(mixed_complex_spec, estimated_noise_mask)
        noise_covariance_matrix = condition_covariance(noise_covariance_matrix, 1e-6) # これがないと性能が大きく落ちる（雑音の共分散行列のみ）
        # ビームフォーマによる雑音除去を実行
        if args.beamformer_type == 'MVDR':
            # target_steering_vectors = estimate_steering_vector(target_covariance_matrix)
            # estimated_spec = mvdr_beamformer(mixed_complex_spec, target_steering_vectors, noise_covariance_matrix)
            estimated_spec = mvdr_beamformer(mixed_complex_spec, target_covariance_matrix, noise_covariance_matrix)
        elif args.beamformer_type == 'GEV':
            estimated_spec = gev_beamformer(mixed_complex_spec, target_covariance_matrix, noise_covariance_matrix)
        elif args.beamformer_type == "DS":
            target_steering_vectors = estimate_steering_vector(target_covariance_matrix)
            estimated_spec = ds_beamformer(mixed_complex_spec, target_steering_vectors)
        elif args.beamformer_type == "MWF":
            estimated_spec = mwf(mixed_complex_spec, target_covariance_matrix, noise_covariance_matrix)
        elif args.beamformer_type == 'Sparse':
            estimated_spec = sparse(mixed_complex_spec, estimated_target_mask) # マスクが正常に推定できているかどうかをテストする用
        else:
            print("Please specify the correct beamformer type")
        """estimated_spec: (num_channels, freq_bins, time_frames=blocksize)"""
        # マルチチャンネルスペクトログラムを音声波形に変換
        multichannel_estimated_voice_data= np.zeros(indata.shape, dtype='float64') # マルチチャンネル音声波形を格納する配列
        # 1chごとスペクトログラムを音声波形に変換
        for i in range(estimated_spec.shape[0]):
            # estimated_voice_data = spec_to_wave(estimated_spec[i, :, :], hop_length)
            estimated_voice_data = librosa.core.istft(estimated_spec[i, :, :], hop_length=args.hop_length)
            multichannel_estimated_voice_data[:, i] = estimated_voice_data
        """multichannel_estimated_voice_data: (num_samples, num_channels)"""
        multichannel_estimated_voice_data = multichannel_estimated_voice_data * 5 # 音量調整のため仮設定 TODO
        q.put(multichannel_estimated_voice_data)

    # 雑音除去を行わない場合
    else:
        q.put(indata.copy())


if __name__ == "__main__":

    # コマンドライン引数を受け取る
    parser = argparse.ArgumentParser(description='Real time voice separation')
    parser.add_argument('-dm', '--denoising_mode', action='store_true', help='whether model denoises audio or not')
    parser.add_argument('-rt', '--record_time', type=int_or_str, default=None, help="recording time[sec]")
    parser.add_argument('-d', '--device', type=int_or_str, default=0, help='input device (numeric ID or substring)')
    parser.add_argument('-mg', '--mic_gain', type=int, default=1, help='Microphone gain')
    parser.add_argument('-sr', '--sample_rate', type=int, default=16000, help='sampling rate')
    parser.add_argument('-al', '--audio_length', type=int, default=3, help='audio length of mask estimator and beamformer input')
    parser.add_argument('-c', '--channels', type=int, default=8, help='number of input channels')
    parser.add_argument('-fs', '--fft_size', type=int, default=512, help='size of fast fourier transform')
    parser.add_argument('-hl', '--hop_length', type=int, default=160, help='number of audio samples between adjacent STFT columns')
    parser.add_argument('-mt', '--model_type', type=str, default='Unet', help='type of mask estimator model')
    parser.add_argument('-bt', '--beamformer_type', type=str, default='MVDR', help='type of beamformer (DS or MVDR or GEV or MWF)')
    parser.add_argument('-tac', '--target_aware_channel', type=int, default=0, help='microphone channel near target source')
    parser.add_argument('-nac', '--noise_aware_channel', type=int, default=4, help='microphone channel near noise source')
    args = parser.parse_args()

    wave_dir = "./audio_data/rec/"
    os.makedirs(wave_dir, exist_ok=True)
    estimated_voice_path = os.path.join(wave_dir, "record.wav")
    # ネットワークモデルと学習済みのパラメータを保存したチェックポイントファイルのパスを指定
    if args.model_type == 'BLSTM':
        model = BLSTMMaskEstimator()
        checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_BLSTM_1201/ckpt_epoch70.pt" # BLSTM small
    elif args.model_type == 'FC':
        model = FCMaskEstimator()
        checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_all_FC_1202/ckpt_epoch80.pt" # FC small
    elif args.model_type == 'Unet':
        model = UnetMaskEstimator_kernel3()
        checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_Unet_aware_1208/ckpt_epoch110.pt" # U-Net aware channel
        pass
    # 前処理クラスのインスタンスを作成
    transform = AudioProcess(args.sample_rate, args.fft_size, args.hop_length, args.model_type)
    # GPUが使える場合はGPUを使用、使えない場合はCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：" , device)
    # 学習済みのパラメータをロード
    model_params = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_params['model_state_dict'])
    # Unetを使って推論
    # ネットワークを推論モードへ
    model.eval()

    # 参考： 「https://python-sounddevice.readthedocs.io/en/0.3.12/examples.html#recording-with-arbitrary-duration」
    q = queue.Queue() # データを格納した順に取り出すことができるキューを作成
    # Make sure the file is opened before recording anything
    with sf.SoundFile(estimated_voice_path, mode='w', samplerate=args.sample_rate,
                      channels=args.channels) as file:
        with sd.InputStream(samplerate=args.sample_rate, blocksize=args.audio_length*args.sample_rate, device=args.device,
                            channels=args.channels, callback=audio_callback):
            print('#' * 50)
            print('press Ctrl+C to stop the recording')
            print('#' * 50)
            while True:
                file.write(q.get())
