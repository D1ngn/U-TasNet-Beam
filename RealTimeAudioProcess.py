# -*- coding:utf-8 -*-

# 必要モジュールのimport
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

from models import MCDUnet_kernel3
from utils import wave_to_spec, spec_to_wave, save_audio_file

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

# 人の声のスペクトログラムを抽出
def extract_voice_spec(model, mixed_amp, mixed_phase):
    """mixed_amp: (channels=8, freq_bins=257, time_steps=301)"""
    """mixed_phase: (channels=8, freq_bins=257, time_steps=301)"""
    # スペクトログラムを正規化
    max_amp = mixed_amp.max()
    normed_mixed_amp = mixed_amp / max_amp
    # モデルの入力サイズに合わせてタイムステップ数を513にパディング
    amp_padded = np.pad(normed_mixed_amp, [(0, 0), (0, 0), (0, 513-mixed_amp.shape[2])], 'constant')
    """amp_padded: (channels=8, freq_bins=257, time_steps=513)"""
    # 0次元目と1次元目に次元を追加
    amp_expanded = amp_padded[np.newaxis, :, :, :]
    """amp_padded: (batch_size=1, channels=8, freq_bins=257, time_steps=513)"""
    # numpy形式のデータをpytorchのテンソルに変換
    amp_tensor = torch.from_numpy(amp_expanded.astype(np.float32)).clone()
    # 環境音のmaskを計算
    mask = model(amp_tensor)
    # pytorchのtensorをnumpy配列に変換
    mask = mask.detach().numpy().copy()
    # 人の声を取り出す
    normed_separated_voice_amp = mask * amp_expanded
    # 正規化によって小さくなった音量を元に戻す
    separated_voice_amp = normed_separated_voice_amp * max_amp
    """separated_voice_amp: (batch_size=1, channels=8, freq_bins=257, time_steps=513)"""
    separated_voice_amp = np.squeeze(separated_voice_amp)
    """separated_voice_amp: (channels=8, freq_bins=257, time_steps=513)"""
    # 入力と同じ大きさのスペクトログラムに戻す
    separated_voice_amp = separated_voice_amp[:, :, :mixed_amp.shape[2]]
    """voice_spec: (channels=8, freq_bins=257, time_steps=301)"""
    # マスクした後の振幅スペクトログラムに入力音声の位相スペクトログラムを掛け合わせて音声を復元
    voice_spec = separated_voice_amp * mixed_phase
    """voice_spec: (channels=8, freq_bins=257, time_steps=301)"""
    return voice_spec

# マイクロホンで取得した音声を固定時間に分割し、処理を行う
def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # マイクロホンのゲイン調整
    indata = indata * args.mic_gain
    # 雑音除去を行う場合
    if args.denoising_mode:
        # マルチチャンネルオーディオデータをスペクトログラムに変換
        multichannel_amp_spec = [] # それぞれのチャンネルの振幅スペクトログラムを格納するリスト
        multichannel_phase_spec = [] # それぞれのチャンネルの位相スペクトログラムを格納するリスト
        # Fancy indexing with mapping creates a (necessary!) copy:
        audio_data = np.asfortranarray(indata.copy())
        for i in range(audio_data.shape[1]):
            audio_amp, audio_phase = wave_to_spec(audio_data[:, i], args.fft_size, args.hop_length)
            multichannel_amp_spec.append(audio_amp)
            multichannel_phase_spec.append(audio_amp)
        multichannel_amp_spec = np.array(multichannel_amp_spec)
        multichannel_phase_spec = np.array(multichannel_phase_spec)
        # マイクで取得した音声のスペクトログラムから人の声のスペクトログラムを抽出
        voice_spec = extract_voice_spec(model, multichannel_amp_spec, multichannel_phase_spec)
        """voice_spec: (num_channels=8, freq_bins=257, time_frames=blocksize)"""
        # マルチチャンネルスペクトログラムを音声波形に変換
        multichannel_estimated_voice_data= np.zeros(audio_data.shape, dtype='float32') # マルチチャンネル音声波形を格納する配列
        for i in range(voice_spec.shape[0]):
            # 1chごとスペクトログラムを音声波形に変換
            estimated_voice_data = spec_to_wave(voice_spec[i, :, :], args.hop_length)
            multichannel_estimated_voice_data[:, i] = estimated_voice_data
        """multichannel_estimated_voice_data: (num_samples, num_channels=8)"""
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
    parser.add_argument('-c', '--channels', type=int, default=8, help='number of input channels')
    parser.add_argument('-fs', '--fft_size', type=int, default=512, help='size of fast fourier transform')
    parser.add_argument('-hl', '--hop_length', type=int, default=160, help='number of audio samples between adjacent STFT columns')
    args = parser.parse_args()

    wave_dir = "./audio_data/rec/"
    os.makedirs(wave_dir, exist_ok=True)
    estimated_voice_path = os.path.join(wave_dir, "record.wav")
    # 学習済みのパラメータを保存したチェックポイントファイルのパスを指定
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_fft_512_kernel3_multi_1007/ckpt_epoch200.pt" # 英語版
    checkpoint_path = "./ckpt/ckpt_JVS_fft_512_kernel3_multi_all_1109/ckpt_epoch210.pt" # 日本語版
    # ネットワークモデルを指定
    model = MCDUnet_kernel3()
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
        with sd.InputStream(samplerate=args.sample_rate, blocksize=48000, device=args.device,
                            channels=args.channels, callback=audio_callback):
            print('#' * 50)
            print('press Ctrl+C to stop the recording')
            print('#' * 50)
            while True:
                file.write(q.get())
