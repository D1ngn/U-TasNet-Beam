# -*- coding:utf-8 -*-

# 必要モジュールのimport
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import os
import numpy as np
import pyaudio
import wave
import struct
import librosa
import argparse
import time

from models import MCDUnet_kernel3
from utils import wave_to_spec, spec_to_wav, save_audio_file


def write_wave(idx, data, sr, save_dir): # sr:サンプリング周波数
    sin_wave = [int(x * 32767.0) for x in data]
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)
    wav_file_path = os.path.join(save_dir, str(idx)+'.wav')
    w = wave.Wave_write(wav_file_path)
    p = (1, 2, sr, len(binwave), 'NONE', 'not compressed')
    w.setparams(p)
    w.writeframes(binwave)
    w.close()


# 人の声のスペクトログラムを抽出
def extract_voice_spec(model, mixed_amp, mixed_phase):
    """mixed_amp: (channels=8, freq_bins=257, time_steps=301)"""
    """mixed_phase: (channels=8, freq_bins=257, time_steps=301)"""
    # スペクトログラムを正規化
    max_amp = mixed_amp.max()
    normed_mixed_amp = mixed_amp / max_amp
    # モデルの入力サイズに合わせてタイムステップ数を513にパディング
    print("normed_mixed_amp.shape:", normed_mixed_amp.shape)
    amp_padded = np.pad(normed_mixed_amp, [(0, 0), (0, 0), (0, 513-mixed_amp.shape[2])], 'constant')
    print("amp_padded.shape:", amp_padded.shape)
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



if __name__ == "__main__":

    # コマンドライン引数を受け取る
    parser = argparse.ArgumentParser(description='Real time voice separation')
    parser.add_argument('-r', '--record_time', default=None, help="Please specify record time[sec]")
    args = parser.parse_args()

    # 使用するパラメータの設定
    N = 10
    CHUNK = 1024 * N # １度に処理する音声のサンプル数
    RATE = 16000 # サンプリングレート
    FORMAT = pyaudio.paInt16
    CHANNELS = 8
    fft_size = 512 # 高速フーリエ変換のフレームサイズ
    hop_length = 160 # 高速フーリエ変換におけるフレーム間のオーバーラップ長
    audio_idx = 0
    wave_dir = "./audio_data/rec/"
    os.makedirs(wave_dir, exist_ok=True)

    # 学習済みのパラメータを保存したチェックポイントファイルのパスを指定
    # checkpoint_path = "./ckpt/ckpt_0509/ckpt_epoch700.pt"
    # checkpoint_path = "./ckpt/ckpt_voice1000_0715/ckpt_epoch10.pt"
    checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_fft_512_kernel3_multi_1007/ckpt_epoch200.pt"
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

    p = pyaudio.PyAudio()

    stream = p.open(format = FORMAT,
    		channels = CHANNELS,
    		rate = RATE,
    		frames_per_buffer = CHUNK,
    		input = True,
    		output = False) # inputとoutputを同時にTrueにする


    if args.record_time == None:
        # マイクで取得した音声に対して音源分離処理を行う
        while stream.is_active():
            # print(audio_idx)
            input = stream.read(CHUNK)
            # input = voice_separate(input) # 音源分離処理を行う
            audio_data = np.frombuffer(input, dtype="int16") / 32768.0 # 量子化ビット数が16bitの場合の正規化
            # マルチチャンネルで行う場合
            # チェンネル1のデータはmultichannel_data[:, 0]、チャンネル2のデータはmultichannel_data[:, 1]...
            # chunk_length = len(data) / CHANNELS
            # multichannel_data = np.reshape(data, (chunk_length, CHANNELS))
            start_time = time.perf_counter()
            # 音声データをスペクトログラムに変換
            multichannel_amp_spec = [] # それぞれのチャンネルの振幅スペクトログラムを格納するリスト
            multichannel_phase_spec = [] # それぞれのチャンネルの位相スペクトログラムを格納するリスト
            for i in range(CHANNELS):
                # オーディオデータをスペクトログラムに変換
                audio_amp, audio_phase = wave_to_spec(audio_data[:, i], fft_size, hop_length)
                multichannel_amp_spec.append(audio_amp)
                multichannel_phase_spec.append(audio_amp)
            multichannel_amp_spec = np.array(multichannel_amp_spec)
            multichannel_phase_spec = np.array(multichannel_phase_spec)

            # mixed_amp, mixed_phase = wave_to_spec(data, fft_size, hop_length) # wavをスペクトログラムへ
            # マイクで取得した音声のスペクトログラムから人の声のスペクトログラムを抽出
            voice_spec = extract_voice_spec(model, mulichannel_amp_spec, mulichannel_phase_spec)
            # スペクトログラムを音声データに変換
            masked_voice_data = spec_to_wav(voice_spec, hop_length)
            finish_time = time.perf_counter()
            print("処理時間：", finish_time - start_time)
            # 音声データを保存
            masked_voice_path = "./output/test/masked_voice{}.wav".format(audio_idx)
            save_audio_file(masked_voice_path, masked_voice_data, sampling_rate=RATE)
            # data =[]
            # data = np.frombuffer(input, dtype="int16") / 32768.0 # 量子化ビット数が16bitの場合の正規化
            # write_wave(audio_idx, data, RATE, wave_dir) # 処理後の音声データをwavファイルとして保存
            # output = stream.write(input)
            audio_idx += 1
    else:
        # マイクで取得した音声に対して音源分離処理を行い、録音する
        print("録音開始")
        all_data = np.empty(0) # 音声データを格納するnumpyの空配列を用意
        np.seterr(divide='ignore', invalid='ignore') # 録音開始時に出る警告を無視
        for i in range(0, int(RATE / CHUNK * float(args.record_time))):
            input = stream.read(CHUNK)
            audio_data = np.frombuffer(input, dtype="int16") / 32768.0 # 量子化ビット数が16bitの場合の正規化
            # 音声データをスペクトログラムに変換
            multichannel_amp_spec = [] # それぞれのチャンネルの振幅スペクトログラムを格納するリスト
            multichannel_phase_spec = [] # それぞれのチャンネルの位相スペクトログラムを格納するリスト
            audio_data = np.asfortranarray(audio_data.reshape([CHUNK, CHANNELS]))
            for j in range(CHANNELS):
                # オーディオデータをスペクトログラムに変換
                audio_amp, audio_phase = wave_to_spec(audio_data[:, j], fft_size, hop_length)
                multichannel_amp_spec.append(audio_amp)
                multichannel_phase_spec.append(audio_amp)
            multichannel_amp_spec = np.array(multichannel_amp_spec)
            multichannel_phase_spec = np.array(multichannel_phase_spec)
            # マイクで取得した音声のスペクトログラムから人の声のスペクトログラムを抽出
            voice_spec = extract_voice_spec(model, multichannel_amp_spec, multichannel_phase_spec)
            # スペクトログラムを音声データに変換
            masked_voice_data = spec_to_wav(voice_spec, hop_length)
            all_data = np.append(all_data, masked_voice_data)

        print("{}秒間録音を行いました".format(str(args.record_time)))
        stream.stop_stream()
        stream.close()
        p.terminate()

        # wavファイルに保存
        masked_voice_path = "./output/rec/record.wav"
        save_audio_file(masked_voice_path, all_data, sampling_rate=RATE)
