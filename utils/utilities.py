# 必要モジュールのimport
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import librosa
import soundfile as sf
import subprocess
import time

from sklearn.linear_model import LinearRegression

# データの前処理を行うクラス
class AudioProcess():
    def __init__(self, sample_rate, fft_size, hop_length, channel_select_type='single', padding=True, num_mels=40):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.channel_select_type = channel_select_type # 'single' or 'median' or 'aware'
        self.padding = padding # True or False
        # ログメルスペクトログラムを使用する場合
        self.num_mels = num_mels
        # U-Netを使用する場合
        self.spec_time_frames = 513
    
    # データを標準化（平均0、分散1に正規化（Z-score Normalization））
    def standardize(self, data):
        data_mean = data.mean(keepdims=True)
        data_std = data.std(keepdims=True, ddof=0) # 母集団の標準偏差（標本標準偏差を使用する場合はddof=1）
        standardized_data = (data - data_mean) / data_std
        return standardized_data
    
    # マルチチャンネルの複素スペクトログラムを算出
    def calc_complex_spec(self, wav_data):
        """wav_data: (num_samples, num_channels)"""
        wav_data = np.require(wav_data, dtype=np.float32, requirements=['F']) # Fortran-contiguousに変換（これがないとエラーが出る）
        multi_complex_spec = [] # それぞれのチャンネルの複素スペクトログラムを格納するリスト
        for i in range(wav_data.shape[1]):
            # オーディオデータをスペクトログラムに変換
            complex_spec = librosa.core.stft(wav_data[:, i], n_fft=self.fft_size, hop_length=self.hop_length, win_length=None, window='hann')
            multi_complex_spec.append(complex_spec)
        multi_complex_spec = np.array(multi_complex_spec)
        """multi_complex_spec: (num_channels, freq_bins, time_frames)"""
        return multi_complex_spec

    # 振幅スペクトログラムを算出
    def calc_amp_spec(self, complex_spec):
        """complex_spec: (num_channels, freq_bins, time_frames)"""
        amp_spec = np.zeros_like(complex_spec, dtype=np.float32)
        for i in range(complex_spec.shape[0]):
            amp_spec[i] = np.abs(complex_spec[i])
        """amp_spec: (num_channels, freq_bin, time_frames)"""
        return amp_spec
    
    # 位相スペクトログラムを算出
    def calc_phase_spec(self, complex_spec):
        """complex_spec: (num_channels, freq_bins, time_frames)"""
        phase_spec = np.zeros_like(complex_spec, dtype=np.float32)
        for i in range(complex_spec.shape[0]):
            phase_spec[i] = np.angle(complex_spec[i]) # ラジアン単位（-π ~ +π）
        """amp_spec: (num_channels, freq_bin, time_frames)"""
        return phase_spec
    
    # ログスペクトログラムを算出
    def calc_log_spec(self, complex_spec):
        """
        complex_spec: (num_channels, freq_bins, time_frames)
        """
        log_spec = np.zeros_like(complex_spec, dtype='float32')
        # チャンネルごとにログスペクトログラムに変換
        for i in range(complex_spec.shape[0]):
            power_spec = np.abs(complex_spec[i]) ** 2 # パワースペクトログラムを算出
            log_spec[i] = 10.0 * np.log10(np.maximum(1e-10, power_spec)) # logがマイナス無限にならないように対数変換
        """log_spec: (num_channels, num_mels, time_frames)"""
        return log_spec
    
    # ログメルスペクトログラムを算出
    def calc_log_mel_spec(self, complex_spec):
        """
        complex_spec: (num_channels, freq_bins, time_frames)
        """
        # メルフィルタバンクを生成
        mel_fb = librosa.filters.mel(self.sample_rate, n_fft=self.fft_size, n_mels=self.num_mels)
        log_mel_spec = np.zeros((complex_spec.shape[0], self.num_mels, complex_spec.shape[2]), dtype='float32')
        # チャンネルごとにログメルスペクトログラムに変換
        for i in range(complex_spec.shape[0]):
            power_spec = np.abs(complex_spec[i]) ** 2 # パワースペクトログラムを算出
            mel_power_spec = np.dot(mel_fb, power_spec) # メルフィルタバンクをかける
            # log_mel_spec[i] = 10.0 * np.log10(np.maximum(1e-10, mel_power_spec)) # logがマイナス無限にならないように対数変換
            log_mel_spec[i] = np.log10(np.maximum(1e-10, mel_power_spec)) # logがマイナス無限にならないように対数変換（VoiceFilterでの計算式に合わせる）
        """log_mel_spec: (num_channels, num_mels, time_frames)"""
        return log_mel_spec
    
    # マルチチャンネルスペクトログラムを音声波形に変換
    def spec_to_wave(self, multi_channel_spec, original_audio_data):
        """
        multi_channel_spec: マルチチャンネルスペクトログラム (num_channels, freq_bins, time_frames)
        original_audio_data: 長さの基準となる音声データ (num_channels, num_samples)
        """
        multi_channel_audio_data = np.zeros(original_audio_data.shape, dtype='float32') # マルチチャンネル音声波形を格納する配列
        # 1chごとスペクトログラムを音声波形に変換
        for i in range(multi_channel_spec.shape[0]):
            estimated_voice_data = librosa.core.istft(multi_channel_spec[i, :, :], hop_length=self.hop_length)
            # 0でパディングして長さを揃える TODO
            estimated_voice_data = np.pad(estimated_voice_data, [0, original_audio_data.shape[0] - estimated_voice_data.shape[0]], "constant")
            multi_channel_audio_data[:, i] = estimated_voice_data
        """multichannel_audio_data: (num_samples, num_channels)"""
        return multi_channel_audio_data
    
    # 話者分離
    # 現在は1ch入力しかサポートしていない → 8ch分まとめて音源分離できるようにする TODO
    def speaker_separation(self, separation_model, mixed_speech_data):
        """
        separation_model: 話者分離用のモデル（Conv-TasNet, Sepformer etc...）
        mixed_speech_data: 複数の話者の発話が混ざった音声 (num_samples, num_channels=1)
        """
        # # 1ch分を取り出す
        # mixed_speech_data = mixed_speech_data[:, 0][:, np.newaxis]
        # """mixed_speech_data: (num_samples, num_channels=1)"""
        mixed_speech_data = mixed_speech_data.transpose()
        """mixed_speech_data: (num_channels=1, num_samples)"""
        mixed_speech_data = mixed_speech_data.reshape(1, mixed_speech_data.shape[0], mixed_speech_data.shape[1])
        """mixed_speech_data: (batch_size=1, num_channels, num_samples)"""
        separated_audio_data = separation_model.separate(mixed_speech_data)
        """separated_audio_data: (num_channels, num_sources, num_samples)"""
        separated_audio_data = separated_audio_data.transpose(1, 2, 0)
        """separated_audio_data: (num_sources, num_samples, num_channels)"""
        return separated_audio_data

    # 分離された発話のうちどれが目的話者の発話かを判断
    def speaker_selector(self, embedder, multiple_speech_data, ref_dvec, device='cpu'):
        """
        embedder: 話者識別モデル
        multiple_speech_data: 複数の発話を含んだ音声データ (num_sources, num_samples, num_channels)
        ref_dvec: 目的話者の発話サンプルの埋め込みベクトル (embed_dim=256,)
        """
        # 話者の埋め込みベクトルを用いて目的の話者の声を選出
        # コサイン類似度の最大値を初期化
        max_cos_similarity = 0
        # 目的話者のIDを初期化
        target_speaker_id = 0
        for speaker_id, speech_data in enumerate(multiple_speech_data):
            speech_complex_spec = self.calc_complex_spec(speech_data)
            """speech_complex_spec: (num_channels, freq_bins, time_frames)"""
            speech_amp_spec = self.calc_amp_spec(speech_complex_spec)
            """speech_amp_spec: (num_channels, freq_bins, time_frames)"""
            # 振幅スペクトログラムをまとめる
            if speaker_id == 0:
                speech_amp_spec_all = speech_amp_spec[np.newaxis, :, :, :]
            else:
                speech_amp_spec_all = np.concatenate([speech_amp_spec_all, speech_amp_spec[np.newaxis, :, :, :]], axis=0)
            """speech_amp_spec_all: (num_sources, num_channels, freq_bins, time_frames)"""
            # 発話の音声波形をログメルスペクトログラムに変換
            speech_log_mel_spec = self.calc_log_mel_spec(speech_complex_spec)
            """speech_log_mel_spec: (num_channels, num_mels, time_frames)"""
            # numpy配列からPyTorchのテンソルに変換
            speech_log_mel_spec = torch.from_numpy(speech_log_mel_spec.astype(np.float32)).clone().to(device)
            # 発話の特徴量をベクトルに変換
            speech_dvec = embedder(speech_log_mel_spec[0]) # 入力は1ch分
            """speech_dvec: (embed_dim=256,)"""
            # 分離音の埋め込みベクトルと発話サンプルの埋め込みベクトルのコサイン類似度を計算
            cos_similarity = F.cosine_similarity(speech_dvec, ref_dvec, dim=0)
            # コサイン類似度が最大となる発話を目的話者の発話と判断
            if max_cos_similarity < cos_similarity:
                max_cos_similarity = cos_similarity
                target_speaker_id = speaker_id
        return target_speaker_id, speech_amp_spec_all
    
    # 過去のマイクロホン入力信号の算出
    def get_past_signal(self, complex_spec, delay, taps):
        """
        complex_spec: マイクロホン入力信号 (num_microphones, freq_bins, time_frames)
        delay: 遅延フレーム数
        taps: 残響除去フィルタのタップ長
        complex_spec_past: 過去のマイクロホン入力信号 (tap_length, M, Nk, Lt)
        """
        # 過去のマイクロホン入力信号の配列を準備
        complex_spec_past = np.zeros(shape=(taps,) + np.shape(complex_spec), dtype=np.complex)
        """complex_spec_past: (tap_length, num_microphones=2, freq_bins=513, time_frames=276)"""
        for tau in range(taps):
            complex_spec_past[tau, :, :, tau+delay:] = complex_spec[:, :, :-(tau+delay)]
        """complex_spec_past: (tap_length, num_microphones, freq_bins, time_frames)"""
        return complex_spec_past
    
    # WPEによるマルチチャンネル音声の残響除去
    def dereverberation_wpe_multi(self, complex_spec, delay=3, taps=10, wpe_iterations=1):
        """
        complex_spec: マイクロホン入力信号 (num_microphones, freq_bins, time_frames)
        wpe_iterations: パラメータの更新回数（リアルタイム処理する場合は1、精度を求めたければ3くらいがベストかも）
        complex_spec_dereverb: 残響除去後の信号 (freq_bins, time_frames) 
        """
        # 過去のマイクロホン入力信号を算出
        complex_spec_past = self.get_past_signal(complex_spec, delay, taps)
        """complex_spec_past: (tap_length, num_microphones, freq_bins, time_frames)"""
        # マイクロホン数・周波数・フレーム数・タップ長を取得する
        M = np.shape(complex_spec)[0]
        Nk = np.shape(complex_spec)[1]
        Lt = np.shape(complex_spec)[2]
        taps = np.shape(complex_spec_past)[0]
        # 過去のマイクロホン入力信号の形式を変更
        x_bar = np.reshape(complex_spec_past, [taps*M, Nk, Lt])
        """x_bar: (num_taps*num_microphones, freq_bins, time_frames)"""
        # 各時間周波数における音声の分散を算出
        var = np.mean(np.square(np.abs(complex_spec)), axis=0)
        var = np.maximum(var, 1.e-8) # 0除算を避けるため
        # var = compute_lambda(complex_spec) # あるいはこれ
        """var: (freq_bins, time_frames)"""
        cost_buff = []
        # 残響除去処理のループ
        for t in range(wpe_iterations):
            x_bar_var_inv = x_bar / var[np.newaxis, :, :]
            """x_bar_var_inv: (num_taps*num_microphones, freq_bins, time_frames)"""
            # 共分散行列を計算
            x_bar_x_bar_h = np.einsum('ift,jft->fij', x_bar_var_inv, np.conjugate(x_bar))
            """x_bar_x_bar_h: (freq_bins, num_taps*num_microphones, num_taps*num_microphones)"""
            # 相関ベクトルを計算
            correlation = np.einsum('ift,mft->fim', x_bar_var_inv, np.conjugate(complex_spec)) # TODO 
            """correlation: (freq_bins, num_taps*num_microphones, num_microphones)"""
            # # 残響除去フィルタを算出（A行列が特異行列（行列式が0となる行列）となり、逆行列が存在しない場合にエラーが出る）
            # dereverb_filter = np.linalg.solve(x_bar_x_bar_h, correlation)
            # 残響除去フィルタを算出（逆行列を確実に計算できるようにする）
            dereverb_filter = stable_solve(x_bar_x_bar_h, correlation)
            """dereverb_filter: (freq_bins, num_taps*num_microphones, num_microphones)"""
            # 残響除去実施
            complex_spec_reverb = np.einsum('fjm,jft->mft', np.conjugate(dereverb_filter), x_bar)
            """complex_spec_reverb: (num_microphones, freq_bins, time_frames)"""
            complex_spec_dereverb = complex_spec - complex_spec_reverb
            """complex_spec_dereverb: (num_microphones, freq_bins, time_frames)"""
            # パラメータ更新
            var = np.mean(np.square(np.abs(complex_spec)), axis=0)
            var = np.maximum(var, 1.e-8) # 0除算を避けるため
            # コスト計算
            cost = np.mean(np.log(var))
            cost_buff.append(cost)
            # # 消しすぎた分
            # complex_spec_dereverb *= 3
        return complex_spec_dereverb, cost_buff
    
    # 音声をミニバッチに分けながら振幅スペクトログラムに変換（サンプル数から1を引いているのは2バッチ目以降のサンプル数が0になるのを防ぐため）
    def preprocess_mask_estimator(self, audio_data, batch_length, device='cpu'):
        batch_size = (audio_data.shape[0] - 1) // batch_length + 1 # （例） 3秒（48000サンプル）ごとに分ける場合72000のだとバッチサイズは2
        for batch_idx in range(batch_size):
            # 音声をミニバッチに分ける
            audio_data_partial = audio_data[batch_length*batch_idx:batch_length*(batch_idx+1), :]
            """audio_data_partial: (num_samples, num_channels)"""
            # マルチチャンネル音声データを複素スペクトログラムと振幅スペクトログラムに変換
            _, amp_spec_partial = self.__call__(audio_data_partial)
            """amp_spec_partial: (num_channels, freq_bins, time_frames)"""
            # 振幅スペクトログラムを標準化
            amp_spec_partial = self.standardize(amp_spec_partial)
            # numpy形式のデータをpytorchのテンソルに変換
            amp_spec_partial = torch.from_numpy(amp_spec_partial.astype(np.float32)).clone().to(device)
            # 振幅スペクトログラムをバッチ方向に結合
            if batch_idx == 0:
                amp_spec_batch = amp_spec_partial.unsqueeze(0)
            else:
                # paddingがFalseの場合、別途長さを揃える必要があるためパディングする
                if self.padding:
                    amp_spec_batch = torch.cat((amp_spec_batch, amp_spec_partial.unsqueeze(0)), dim=0)
                else:
                    pad = nn.ZeroPad2d((0, amp_spec_batch.shape[3]-amp_spec_partial.shape[2], 0, 0))
                    amp_spec_batch = torch.cat((amp_spec_batch, pad(amp_spec_partial.unsqueeze(0))), dim=0)
        """amp_spec_batch: (batch_size, num_channels, freq_bins, time_frames)"""
        return amp_spec_batch
    
    # ミニバッチに分けられたマスクを時間方向に結合し、混合音にかけて各音源のスペクトログラムを取得
    def postprocess_mask_estimator(self, mixed_complex_spec, mask, batch_length, aware_channel=0):
        """
        mixed_complex_spec: 混合音のスペクトログラム (num_channels, freq_bins, time_frames)
        mask: 各音源の時間周波数マスク (batch_size, num_channels, freq_bins, time_frames)
        aware_channel: チャンネルの選択方式がawareの場合に選択するチャンネル（medianでは使わない） TODO
        """
        if self.channel_select_type == 'aware':
            # マスクのチャンネルを指定（目的音に近いチャンネルと雑音に近いチャンネル）
            mask = mask[:, aware_channel, :, :]
            """mask: (batch_size, freq_bins, time_frames)"""
        elif self.channel_select_type == 'median':
            # 複数チャンネル間のマスク値の中央値をとる（median pooling）
            (mask, _) = torch.median(mask, dim=1)
            """mask: (batch_size, freq_bins, time_frames)"""
        elif self.channel_select_type == 'single':
            mask = mask[:, 0, :, :]
            """mask: (batch_size, freq_bins, time_frames)"""
        else:
            print("Please specify the correct channel selection type")
        # paddingされた分を削除する
        if self.padding:
            mask = mask[:, :, :int(batch_length/self.hop_length)+1] 
        # バッチ方向に並んだデータを時間方向につなげる
        mask = torch.transpose(mask, 0, 1)
        """mask: (freq_bins, batch_size, time_frames)"""
        mask = mask.contiguous().view(mask.shape[0], -1)
        """mask: (freq_bins, batch_size*time_frames)"""
        # 時間方向の長さを元に戻す
        mask = mask[:, :mixed_complex_spec.shape[2]]
        # pytorchのテンソルをnumpy形式のデータに変換
        mask = mask.to('cpu').detach().numpy().copy() # CPU
        # マスクを混合音声に掛けてスペクトログラムを抽出
        estimated_spec = mask[np.newaxis, :, :] * mixed_complex_spec
        """multichannel_speech_spec: (num_channels, freq_bins, time_frames)"""
        return estimated_spec, mask
    
    def __call__(self, wav_data):
        # マルチチャンネル音声データをスペクトログラムに変換
        multi_complex_spec = self.calc_complex_spec(wav_data)
        """multi_complex_spec: (num_channels, freq_bins, time_frames)"""
        # モデルに入力するデータのパディングを行う場合 
        if self.padding:
            # モデルの入力サイズに合わせて、スペクトログラムの後ろの部分を0埋め(パディング)
            # データが三次元の時、(手前, 奥),(上,下), (左, 右)の順番でパディングを追加
            multi_complex_spec = np.pad(multi_complex_spec, [(0, 0), (0, 0), (0, self.spec_time_frames-multi_complex_spec.shape[2])], 'constant')
            """multi_complex_spec: (num_channels=8, freq_bins=257, time_frames=513)"""
        # 振幅スペクトログラムを算出
        multi_amp_spec = self.calc_amp_spec(multi_complex_spec)
        """multi_amp_spec: (num_channels, freq_bins, time_frames)"""
        # # ログスペクトログラムを算出
        # multi_log_spec = self.calc_log_spec(multi_complex_spec)
        # """multi_log_spec: (num_channels, freq_bins, time_frames)"""
        # # 位相スペクトログラムを算出
        # multi_phase_spec = self.calc_phase_spec(multi_complex_spec)
        # """multi_phase_spec: (num_channels, freq_bins, time_frames)"""
        # # 振幅スペクトログラムの代わりにログメルスペクトログラムを使用する場合 TODO
        # multi_log_mel_spec = self.calc_log_mel_spec(multi_complex_spec)
        # return multi_complex_spec, multi_amp_spec, multi_log_mel_spec
        return multi_complex_spec, multi_amp_spec
        # return multi_complex_spec, multi_log_spec

# データの前処理を行うクラス
class AudioProcessForComplex():
    def __init__(self, sample_rate, fft_size, hop_length, padding=True, num_mels=40):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.padding = padding # True or False
        # ログメルスペクトログラムを使用する場合
        self.num_mels = num_mels
        # U-Netを使用する場合
        self.max_time_frames = 513
    
    # データを標準化（平均0、分散1に正規化（Z-score Normalization））
    def standardize(self, data):
        data_mean = data.mean(keepdims=True)
        data_std = data.std(keepdims=True, ddof=0) # 母集団の標準偏差（標本標準偏差を使用する場合はddof=1）
        standardized_data = (data - data_mean) / data_std
        return standardized_data
    
    # マルチチャンネルの複素スペクトログラムを算出
    def calc_complex_spec(self, wav_data):
        """wav_data: (num_samples, num_channels)"""
        wav_data = np.require(wav_data, dtype=np.float32, requirements=['F']) # Fortran-contiguousに変換（これがないとエラーが出る）
        multi_complex_spec = [] # それぞれのチャンネルの複素スペクトログラムを格納するリスト
        for i in range(wav_data.shape[1]):
            # オーディオデータをスペクトログラムに変換
            complex_spec = librosa.core.stft(wav_data[:, i], n_fft=self.fft_size, hop_length=self.hop_length, win_length=None, window='hann')
            multi_complex_spec.append(complex_spec)
        multi_complex_spec = np.array(multi_complex_spec)
        """multi_complex_spec: (num_channels, freq_bins, time_frames)"""
        return multi_complex_spec

    # 振幅スペクトログラムを算出
    def calc_amp_spec(self, complex_spec):
        """complex_spec: (num_channels, freq_bins, time_frames)"""
        amp_spec = np.zeros_like(complex_spec, dtype=np.float32)
        for i in range(complex_spec.shape[0]):
            amp_spec[i] = np.abs(complex_spec[i])
        """amp_spec: (num_channels, freq_bin, time_frames)"""
        return amp_spec
    
    # 位相スペクトログラムを算出
    def calc_phase_spec(self, complex_spec):
        """complex_spec: (num_channels, freq_bins, time_frames)"""
        phase_spec = np.zeros_like(complex_spec, dtype=np.float32)
        for i in range(complex_spec.shape[0]):
            phase_spec[i] = np.angle(complex_spec[i]) # ラジアン単位（-π ~ +π）
        """amp_spec: (num_channels, freq_bin, time_frames)"""
        return phase_spec
    
    # ログスペクトログラムを算出
    def calc_log_spec(self, complex_spec):
        """
        complex_spec: (num_channels, freq_bins, time_frames)
        """
        log_spec = np.zeros_like(complex_spec, dtype='float32')
        # チャンネルごとにログスペクトログラムに変換
        for i in range(complex_spec.shape[0]):
            power_spec = np.abs(complex_spec[i]) ** 2 # パワースペクトログラムを算出
            log_spec[i] = 10.0 * np.log10(np.maximum(1e-10, power_spec)) # logがマイナス無限にならないように対数変換
        """log_spec: (num_channels, num_mels, time_frames)"""
        return log_spec
    
    # ログメルスペクトログラムを算出
    def calc_log_mel_spec(self, complex_spec):
        """
        complex_spec: (num_channels, freq_bins, time_frames)
        """
        # メルフィルタバンクを生成
        mel_fb = librosa.filters.mel(self.sample_rate, n_fft=self.fft_size, n_mels=self.num_mels)
        log_mel_spec = np.zeros((complex_spec.shape[0], self.num_mels, complex_spec.shape[2]), dtype='float32')
        # チャンネルごとにログメルスペクトログラムに変換
        for i in range(complex_spec.shape[0]):
            power_spec = np.abs(complex_spec[i]) ** 2 # パワースペクトログラムを算出
            mel_power_spec = np.dot(mel_fb, power_spec) # メルフィルタバンクをかける
            # log_mel_spec[i] = 10.0 * np.log10(np.maximum(1e-10, mel_power_spec)) # logがマイナス無限にならないように対数変換
            log_mel_spec[i] = np.log10(np.maximum(1e-10, mel_power_spec)) # logがマイナス無限にならないように対数変換（VoiceFilterでの計算式に合わせる）
        """log_mel_spec: (num_channels, num_mels, time_frames)"""
        return log_mel_spec
    
    # マルチチャンネルスペクトログラムを音声波形に変換
    def spec_to_wave(self, multi_channel_spec, original_audio_data):
        """
        multi_channel_spec: マルチチャンネルスペクトログラム (num_channels, freq_bins, time_frames)
        original_audio_data: 長さの基準となる音声データ (num_channels, num_samples)
        """
        multi_channel_audio_data = np.zeros(original_audio_data.shape, dtype='float32') # マルチチャンネル音声波形を格納する配列
        # 1chごとスペクトログラムを音声波形に変換
        for i in range(multi_channel_spec.shape[0]):
            estimated_voice_data = librosa.core.istft(multi_channel_spec[i, :, :], hop_length=self.hop_length)
            # 0でパディングして長さを揃える TODO
            estimated_voice_data = np.pad(estimated_voice_data, [0, original_audio_data.shape[0] - estimated_voice_data.shape[0]], "constant")
            multi_channel_audio_data[:, i] = estimated_voice_data
        """multichannel_audio_data: (num_samples, num_channels)"""
        return multi_channel_audio_data
    
    # 話者分離
    # 現在は1ch入力しかサポートしていない → 8ch分まとめて音源分離できるようにする TODO
    def speaker_separation(self, separation_model, mixed_speech_data):
        """
        separation_model: 話者分離用のモデル（Conv-TasNet, Sepformer etc...）
        mixed_speech_data: 複数の話者の発話が混ざった音声 (num_samples, num_channels)
        """
#         # 1ch分を取り出す
#         mixed_speech_data = mixed_speech_data[:, 0][:, np.newaxis]
#         """mixed_speech_data: (num_samples, num_channels=1)"""
        mixed_speech_data = mixed_speech_data.transpose()
        """mixed_speech_data: (num_channels=1, num_samples)"""
        mixed_speech_data = mixed_speech_data.reshape(1, mixed_speech_data.shape[0], mixed_speech_data.shape[1])
        """mixed_speech_data: (batch_size=1, num_channels, num_samples)"""
        separated_audio_data = separation_model.separate(mixed_speech_data)
        """separated_audio_data: (num_channels, num_sources, num_samples)"""
        separated_audio_data = separated_audio_data.transpose(1, 2, 0)
        """separated_audio_data: (num_sources, num_samples, num_channels)"""
        return separated_audio_data

    # 分離された発話のうちどれが目的話者の発話かを判断
    def speaker_selector(self, embedder, multiple_speech_data, ref_dvec):
#     def speaker_selector(self, embedder, multiple_speech_data, ref_dvec, interference_azimuth, file_num):
        """
        embedder: 話者識別モデル
        multiple_speech_data: 複数の発話を含んだ音声データ (num_sources, num_samples, num_channels)
        ref_dvec: 目的話者の発話サンプルの埋め込みベクトル (embed_dim=256,)
        """
        # 話者の埋め込みベクトルを用いて目的の話者の声を選出
        # コサイン類似度の最大値を初期化
        max_cos_similarity = 0
        # 目的話者のIDを初期化
        target_speaker_id = 0
        for speaker_id, speech_data in enumerate(multiple_speech_data):
            speech_complex_spec = self.calc_complex_spec(speech_data)
            """speech_complex_spec: (num_channels, freq_bins, time_frames)"""
            speech_amp_spec = self.calc_amp_spec(speech_complex_spec)
            """speech_amp_spec: (num_channels, freq_bins, time_frames)"""
            
#             # 複素スペクトログラムをまとめる
#             if speaker_id == 0:
#                 speech_complex_spec_all = speech_complex_spec[np.newaxis, :, :, :]
#             else:
#                 speech_complex_spec_all = np.concatenate([speech_complex_spec_all, speech_complex_spec[np.newaxis, :, :, :]], axis=0)
#             """speech_complex_spec_all: (num_sources, num_channels, freq_bins, time_frames)"""
            
            # 振幅スペクトログラムをまとめる
            if speaker_id == 0:
                speech_amp_spec_all = speech_amp_spec[np.newaxis, :, :, :]
            else:
                speech_amp_spec_all = np.concatenate([speech_amp_spec_all, speech_amp_spec[np.newaxis, :, :, :]], axis=0)
            """speech_amp_spec_all: (num_sources, num_channels, freq_bins, time_frames)"""    
            # 発話の音声波形をログメルスペクトログラムに変換
            speech_log_mel_spec = self.calc_log_mel_spec(speech_complex_spec)
            """speech_log_mel_spec: (num_channels, num_mels, time_frames)"""
            # numpy配列からPyTorchのテンソルに変換
            speech_log_mel_spec = torch.from_numpy(speech_log_mel_spec.astype(np.float32)).clone()
            # 発話の特徴量をベクトルに変換
            speech_dvec = embedder(speech_log_mel_spec[0]) # 入力は1ch分
            # PyTorchのテンソルからnumpy配列に変換
            speech_dvec = speech_dvec.detach().numpy().copy() # CPU
            """speech_dvec: (embed_dim=256,)"""
            # 分離音の埋め込みベクトルと発話サンプルの埋め込みベクトルのコサイン類似度を計算
            cos_similarity = np.dot(speech_dvec, ref_dvec) / (np.linalg.norm(speech_dvec)*np.linalg.norm(ref_dvec))
            # コサイン類似度が最大となる発話を目的話者の発話と判断
            if max_cos_similarity < cos_similarity:
                max_cos_similarity = cos_similarity
                target_speaker_id = speaker_id
                
#          # テスト用に話者分離後の音声を保存
#         separated_dir = "./separated_voice/{}/".format(interference_azimuth)
#         os.makedirs(separated_dir, exist_ok=True)
#         separated_voice_path = os.path.join(separated_dir, "{}_separated_voice.wav".format(file_num))
#         separated_voice_data = self.spec_to_wave(speech_complex_spec_all[target_speaker_id], multiple_speech_data[target_speaker_id])
#         sf.write(separated_voice_path, separated_voice_data, 16000)
            
        return target_speaker_id, speech_amp_spec_all
    
    # 分離された発話のうちどれが目的話者の発話かを判断
    def speaker_selector_sig_ver(self, multiple_speech_data, ref_dvec, embedder, device):
#     def speaker_selector(self, embedder, multiple_speech_data, ref_dvec, interference_azimuth, file_num):
        """
        multiple_speech_data: 複数の発話を含んだ音声データ (num_speakers, num_samples, num_channels)
        ref_dvec: 目的話者の発話サンプルの埋め込みベクトル (embed_dim=256,)
        embedder: 話者識別モデル
        """
        # 話者の埋め込みベクトルを用いて目的の話者の声を選出
        # コサイン類似度の最大値を初期化
        max_cos_similarity = 0
        # 目的話者のIDを初期化
        target_speaker_id = 0
        for speaker_id, speech_data in enumerate(multiple_speech_data):
            speech_complex_spec = self.calc_complex_spec(speech_data)
            """speech_complex_spec: (num_channels, freq_bins, time_frames)"""
            # 複素スペクトログラムをまとめる
            if speaker_id == 0:
                speech_complex_spec_all = speech_complex_spec[np.newaxis, :, :, :]
            else:
                speech_complex_spec_all = np.concatenate([speech_complex_spec_all, speech_complex_spec[np.newaxis, :, :, :]], axis=0)
            """speech_complex_spec_all: (num_speakers, num_channels, freq_bins, time_frames)"""
            # 発話の音声波形をログメルスペクトログラムに変換
            speech_log_mel_spec = self.calc_log_mel_spec(speech_complex_spec)
            """speech_log_mel_spec: (num_channels, num_mels, time_frames)"""
            # numpy配列からPyTorchのテンソルに変換
            speech_log_mel_spec = torch.from_numpy(speech_log_mel_spec).float().to(device)
            # 発話の特徴量をベクトルに変換
            speech_dvec = embedder(speech_log_mel_spec[0]) # 入力は1ch分
            # PyTorchのテンソルからnumpy配列に変換
            speech_dvec = speech_dvec.cpu().detach().numpy().copy() # CPU
            """speech_dvec: (embed_dim=256,)"""
            # 分離音の埋め込みベクトルと発話サンプルの埋め込みベクトルのコサイン類似度を計算
            cos_similarity = np.dot(speech_dvec, ref_dvec) / (np.linalg.norm(speech_dvec)*np.linalg.norm(ref_dvec))
            # コサイン類似度が最大となる発話を目的話者の発話と判断
            if max_cos_similarity < cos_similarity:
                max_cos_similarity = cos_similarity
                target_speaker_id = speaker_id
            # # 散布図を描画
            # sns.set() # スタイルをきれいにするsns.set() # スタイルをきれいにする
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # plt.scatter(ref_dvec, speech_dvec)
            # plt.xlabel("d-vector of voice sample")
            # plt.ylabel("d-vector of separated speech{}".format(speaker_id+1))
            # plt.grid(True)
            # # 回帰直線を追加
            # # 回帰分析 線形
            # mod = LinearRegression()
            # ref_dvec = pd.DataFrame(ref_dvec)
            # speech_dvec = pd.DataFrame(speech_dvec)
            # # 線形回帰モデル、予測値、R^2を評価
            # mod_lin = mod.fit(ref_dvec, speech_dvec)
            # y_lin_fit = mod_lin.predict(ref_dvec)
            # r2_lin = mod.score(ref_dvec, speech_dvec)
            # ax.text(0.10, 0.10, '$ R^{2} $=' + str(round(r2_lin, 3)), color="red")
            # plt.plot(ref_dvec.values[:,:], y_lin_fit, color = 'red', linewidth=0.5)
            # # グラフをファイルに保存する
            # fig.savefig("./test/correlation_of_separated_and_ref/Correlation of separated speech{} dvec and ref dvec.png".format(speaker_id+1))
        #  # テスト用に話者分離後の音声を保存
        # separated_dir = "./separated_voice/{}/".format(interference_azimuth)
        # os.makedirs(separated_dir, exist_ok=True)
        # separated_voice_path = os.path.join(separated_dir, "{}_separated_voice.wav".format(file_num))
        # separated_voice_data = self.spec_to_wave(speech_complex_spec_all[target_speaker_id], multiple_speech_data[target_speaker_id])
        # sf.write(separated_voice_path, separated_voice_data, 16000)    
        return target_speaker_id, speech_complex_spec_all
    
    # 過去のマイクロホン入力信号の算出
    def get_past_signal(self, complex_spec, delay, taps):
        """
        complex_spec: マイクロホン入力信号 (num_microphones, freq_bins, time_frames)
        delay: 遅延フレーム数
        taps: 残響除去フィルタのタップ長
        complex_spec_past: 過去のマイクロホン入力信号 (tap_length, M, Nk, Lt)
        """
        # 過去のマイクロホン入力信号の配列を準備
        complex_spec_past = np.zeros(shape=(taps,) + np.shape(complex_spec), dtype=np.complex)
        """complex_spec_past: (tap_length, num_microphones=2, freq_bins=513, time_frames=276)"""
        for tau in range(taps):
            complex_spec_past[tau, :, :, tau+delay:] = complex_spec[:, :, :-(tau+delay)]
        """complex_spec_past: (tap_length, num_microphones, freq_bins, time_frames)"""
        return complex_spec_past
    
    # WPEによるマルチチャンネル音声の残響除去
    def dereverberation_wpe_multi(self, complex_spec, delay=3, taps=10, wpe_iterations=1):
        """
        complex_spec: マイクロホン入力信号 (num_microphones, freq_bins, time_frames)
        wpe_iterations: パラメータの更新回数（リアルタイム処理する場合は1、精度を求めたければ3くらいがベストかも）
        complex_spec_dereverb: 残響除去後の信号 (freq_bins, time_frames) 
        """
        # 過去のマイクロホン入力信号を算出
        complex_spec_past = self.get_past_signal(complex_spec, delay, taps)
        """complex_spec_past: (tap_length, num_microphones, freq_bins, time_frames)"""
        # マイクロホン数・周波数・フレーム数・タップ長を取得する
        M = np.shape(complex_spec)[0]
        Nk = np.shape(complex_spec)[1]
        Lt = np.shape(complex_spec)[2]
        taps = np.shape(complex_spec_past)[0]
        # 過去のマイクロホン入力信号の形式を変更
        x_bar = np.reshape(complex_spec_past, [taps*M, Nk, Lt])
        """x_bar: (num_taps*num_microphones, freq_bins, time_frames)"""
        # 各時間周波数における音声の分散を算出
        var = np.mean(np.square(np.abs(complex_spec)), axis=0)
        var = np.maximum(var, 1.e-8) # 0除算を避けるため
        # var = compute_lambda(complex_spec) # あるいはこれ
        """var: (freq_bins, time_frames)"""
        cost_buff = []
        # 残響除去処理のループ
        for t in range(wpe_iterations):
            x_bar_var_inv = x_bar / var[np.newaxis, :, :]
            """x_bar_var_inv: (num_taps*num_microphones, freq_bins, time_frames)"""
            # 共分散行列を計算
            x_bar_x_bar_h = np.einsum('ift,jft->fij', x_bar_var_inv, np.conjugate(x_bar))
            """x_bar_x_bar_h: (freq_bins, num_taps*num_microphones, num_taps*num_microphones)"""
            # 相関ベクトルを計算
            correlation = np.einsum('ift,mft->fim', x_bar_var_inv, np.conjugate(complex_spec)) # TODO 
            """correlation: (freq_bins, num_taps*num_microphones, num_microphones)"""
            # # 残響除去フィルタを算出（A行列が特異行列（行列式が0となる行列）となり、逆行列が存在しない場合にエラーが出る）
            # dereverb_filter = np.linalg.solve(x_bar_x_bar_h, correlation)
            # 残響除去フィルタを算出（逆行列を確実に計算できるようにする）
            dereverb_filter = stable_solve(x_bar_x_bar_h, correlation)
            """dereverb_filter: (freq_bins, num_taps*num_microphones, num_microphones)"""
            # 残響除去実施
            complex_spec_reverb = np.einsum('fjm,jft->mft', np.conjugate(dereverb_filter), x_bar)
            """complex_spec_reverb: (num_microphones, freq_bins, time_frames)"""
            complex_spec_dereverb = complex_spec - complex_spec_reverb
            """complex_spec_dereverb: (num_microphones, freq_bins, time_frames)"""
            # パラメータ更新
            var = np.mean(np.square(np.abs(complex_spec)), axis=0)
            var = np.maximum(var, 1.e-8) # 0除算を避けるため
            # コスト計算
            cost = np.mean(np.log(var))
            cost_buff.append(cost)
            # # 消しすぎた分
            # complex_spec_dereverb *= 3
        return complex_spec_dereverb, cost_buff
    
    # 音声をミニバッチに分けながら振幅スペクトログラムに変換（サンプル数から1を引いているのは2バッチ目以降のサンプル数が0になるのを防ぐため）
    def preprocess_mask_estimator(self, audio_data, batch_length):
        # torch.stftを使う場合
        batch_size = (audio_data.shape[1] - 1) // batch_length + 1 # （例） 3秒（48000サンプル）ごとに分ける場合72000だとバッチサイズは2
        for batch_idx in range(batch_size):
            # 音声をミニバッチに分ける
            audio_data_partial = audio_data[:, batch_length*batch_idx:batch_length*(batch_idx+1)]
            """audio_data_partial: (num_channels, num_samples)"""
            # マルチチャンネル音声データを振幅スペクトログラム＋位相スペクトログラムに変換
            amp_phase_spec_partial = self.__call__(audio_data_partial)
            """amp_phase_spec_partial: (num_channels, freq_bins, time_frames, real_imaginary)"""
            # 振幅スペクトログラムをバッチ方向に結合
            if batch_idx == 0:
                amp_phase_spec_batch = amp_phase_spec_partial.unsqueeze(0)
            else:
                # paddingがFalseの場合、別途長さを揃える必要があるためパディングする
                if self.padding:
                    amp_phase_spec_batch = torch.cat((amp_phase_spec_batch, amp_phase_spec_partial.unsqueeze(0)), dim=0)
                else:
                    pad = nn.ZeroPad2d((0, amp_phase_spec_batch.shape[3]-amp_phase_spec_partial.shape[2], 0, 0))
                    amp_phase_spec_batch = torch.cat((amp_phase_spec_batch, pad(amp_phase_spec_partial.unsqueeze(0))), dim=0)
        """amp_phase_spec_batch: (batch_size, num_channels, freq_bins, time_frames, real_imaginary)"""

#         # librosa.stftを使う場合
#         batch_size = (audio_data.shape[0] - 1) // batch_length + 1 # （例） 3秒（48000サンプル）ごとに分ける場合72000だとバッチサイズは2
#         for batch_idx in range(batch_size):
#             audio_data_partial = audio_data[batch_length*batch_idx:batch_length*(batch_idx+1), :]
#             """audio_data_partial: (num_samples, num_channels)"""
#             # マルチチャンネル音声データを複素スペクトログラムと振幅スペクトログラムに変換
#             amp_phase_spec_partial = self.__call__(audio_data_partial)
#             """amp_phase_spec_partial: (num_channels, freq_bins, time_frames, real_imaginary)"""
#             # numpy形式のデータをpytorchのテンソルに変換
#             amp_phase_spec_partial = torch.from_numpy(amp_phase_spec_partial.astype(np.float32)).clone()
#             # 振幅スペクトログラムをバッチ方向に結合
#             if batch_idx == 0:
#                 amp_phase_spec_batch = amp_phase_spec_partial.unsqueeze(0)
#             else:
#                 # paddingがFalseの場合、別途長さを揃える必要があるためパディングする
#                 if self.padding:
#                     amp_phase_spec_batch = torch.cat((amp_phase_spec_batch, amp_phase_spec_partial.unsqueeze(0)), dim=0)
#                 else:
#                     pad = nn.ZeroPad2d((0, amp_phase_spec_batch.shape[3]-amp_phase_spec_partial.shape[2], 0, 0))
#                     amp_phase_spec_batch = torch.cat((amp_phase_spec_batch, pad(amp_phase_spec_partial.unsqueeze(0))), dim=0)
#         """amp_phase_spec_batch: (batch_size, num_channels, freq_bins, time_frames)"""
        return amp_phase_spec_batch
    
    # ミニバッチに分けられたマスクを時間方向に結合し、混合音にかけて各音源のスペクトログラムを取得
    def postprocess_mask_estimator(self, mixed_complex_spec, amp_phase_spec, batch_length):
        """
        mixed_complex_spec: 混合音のスペクトログラム (num_channels, freq_bins, time_frames)
        amp_phase_spec: 振幅＋位相スペクトログラム (batch_size, num_channels, freq_bins, time_frames, real_imaginary)
        """            
        # paddingされた分を削除する
        if self.padding:
            amp_phase_spec = amp_phase_spec[:, :, :, :int(batch_length/self.hop_length)+1, :]
        """amp_phase_spec: (batch_size, num_channels, freq_bins, time_frames, real_imaginary)""" 
        # バッチ方向に並んだデータを時間方向につなげる
        amp_phase_spec = amp_phase_spec.permute(1, 2, 0, 3, 4)
        """amp_phase_spec: (num_channels, freq_bins, batch_size, time_frames, real_imaginary)""" 
        amp_phase_spec = amp_phase_spec.contiguous().view(amp_phase_spec.shape[0], amp_phase_spec.shape[1], -1, amp_phase_spec.shape[4])
        """amp_phase_spec: (num_channels, freq_bins, batch_size*time_frames, real_imaginary)""" 
        # 時間方向の長さを元に戻す
        amp_phase_spec = amp_phase_spec[:, :, :mixed_complex_spec.shape[2]]
        """amp_phase_spec: (num_channels, freq_bins, batch_size*time_frames, real_imaginary)""" 
        return amp_phase_spec
    
    # マルチチャンネル音声のロード
    def load_audio(self, file_path):
        waveform, _ = torchaudio.load(file_path)
        """waveform: (num_channels, num_samples)"""
        return waveform
    
    # マルチチャンネルの振幅スペクトログラムと位相スペクトログラムを算出
    def calc_amp_phase_spec(self, waveform):
        """waveform: (num_channels, num_samples)"""
        multi_amp_phase_spec = torch.stft(input=waveform, n_fft=self.fft_size, hop_length=self.hop_length, normalized=True, return_complex=False)
        """multi_amp_phase_spec: (num_channels, freq_bins, time_frames, real_imaginary)"""
        return multi_amp_phase_spec
    
    # マルチチャンネルの複素スペクトログラムを時間フーレム方向に0パディング
    def zero_pad_spec(self, complex_spec):
        """complex_spec: (num_channels, freq_bins, time_frames, real-imaginary)"""
        complex_spec_padded = nn.ZeroPad2d((0, 0, 0, self.max_time_frames-complex_spec.shape[2]))(complex_spec)
        """complex_spec: (num_channels, freq_bins, time_frames=self.max_time_frames, real_imaginary)"""
        return complex_spec_padded
    
    def __call__(self, wav_data):
        # torch.stftを使う場合
        # マルチチャンネル音声データを振幅スペクトログラム＋位相スペクトログラムに変換
        multi_amp_phase_spec = self.calc_amp_phase_spec(wav_data)
        """multi_amp_phase_spec: (num_channels, freq_bins, time_frames, real_imaginary)"""
        # モデルに入力するデータのパディングを行う場合 
        if self.padding:
            # モデルの入力サイズに合わせて、スペクトログラムの後ろの部分を0埋め(パディング)
            multi_amp_phase_spec = self.zero_pad_spec(multi_amp_phase_spec)
            """multi_amp_phase_spec: (num_channels=8, freq_bins=257, time_frames=513, real_imaginary=2)"""
#         # librosa.stftを使う場合
#         # マルチチャンネル音声データをスペクトログラムに変換
#         multi_complex_spec = self.calc_complex_spec(wav_data)
#         """multi_complex_spec: (num_channels, freq_bins, time_frames)"""
#         # モデルに入力するデータのパディングを行う場合 
#         if self.padding:
#             # モデルの入力サイズに合わせて、スペクトログラムの後ろの部分を0埋め(パディング)
#             # データが三次元の時、(手前, 奥),(上,下), (左, 右)の順番でパディングを追加
#             multi_complex_spec = np.pad(multi_complex_spec, [(0, 0), (0, 0), (0, self.max_time_frames-multi_complex_spec.shape[2])], 'constant')
#             """multi_complex_spec: (num_channels=8, freq_bins=257, time_frames=513)"""
#         # 振幅スペクトログラムと位相スペクトログラムを算出して結合
#         multi_amp_spec = self.calc_amp_spec(multi_complex_spec)
#         # 振幅スペクトログラムを標準化
#         multi_amp_spec = self.standardize(multi_amp_spec)
#         multi_phase_spec = self.calc_phase_spec(multi_complex_spec)
#         multi_amp_phase_spec = np.concatenate([multi_amp_spec[:, :, :, np.newaxis], multi_phase_spec[:, :, :, np.newaxis]], axis=3)
#         """multi_amp_phase_spec: (num_channels=8, freq_bins=257, time_frames=513, real_imaginary=2)"""
        return multi_amp_phase_spec
    
# WPEの分散計算用（現在は使っていない）
def compute_lambda(dereverb, ctx=0):
    """
    参考：「https://github.com/funcwj/setk/blob/master/scripts/sptk/libs/wpe.py」
    Compute spatial correlation matrix, using scaled identity matrix method
    Arguments:
        dereverb: N x F x T
        ctx: left/right context used to compute lambda
    Returns:
        lambda: F x T
    """
    def cpw(mat):
        return mat.real**2 + mat.imag**2
    # F x T
    L = np.mean(cpw(dereverb), axis=0)
    _, T = L.shape
    counts_ = np.zeros(T)
    lambda_ = np.zeros_like(L)
    for c in range(-ctx, ctx + 1):
        s = max(c, 0)
        e = min(T, T + c)
        lambda_[:, s:e] += L[:, max(-c, 0):min(T, T - c)]
        counts_[s:e] += 1
    return np.maximum(lambda_ / counts_, 1.e-8)


# AX = Bを解き、Xを求める
def stable_solve(A, B):
    """
    参考：「https://github.com/fgnt/nara_wpe/blob/master/nara_wpe/wpe.py」
    Use np.linalg.solve with fallback to np.linalg.lstsq.
    Equal to np.linalg.lstsq but faster.
    Note: limited currently by A.shape == B.shape
    This function try's np.linalg.solve with independent dimensions,
    when this is not working the function fall back to np.linalg.solve
    for each matrix. If one matrix does not work it fall back to
    np.linalg.lstsq.
    The reason for not using np.linalg.lstsq directly is the execution time.
    Examples:
    A and B have the shape (500, 6, 6), than a loop over lstsq takes
    108 ms and this function 28 ms for the case that one matrix is singular
    else 1 ms.
    """
    assert A.shape[:-2] == B.shape[:-2], (A.shape, B.shape)
    assert A.shape[-1] == B.shape[-2], (A.shape, B.shape)
    # Aの逆行列を求めてAX = Bを解く（X = A^-1Bを満たすXを解として求める）
    try:
        return np.linalg.solve(A, B)
    # Aが特異行列である場合は最小二乗法を使用してAX = Bを解く（|B-AX|^2が小さくなるようなXを解として求める）
    # 処理速度が遅いので、周波数ごとにnp.linalg.solveとnp.linalg.lstsqを使い分けた方が良いかも TODO
    except np.linalg.LinAlgError:
        X = np.zeros_like(B)
        for i in range(A.shape[0]):
            X[i] = np.linalg.lstsq(A[i], B[i], rcond=None)[0]
        return X


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

# waveファイルを読み込み波形のグラフを保存する
def wave_plot(input_path, output_path, ylim_min=-1.0, ylim_max=1.0, fig_title=None):
    # 音声をロード
    data, sample_rate = sf.read(input_path)
    # マルチチャンネルの場合全てのチャンネルの平均を取る
    if data.ndim == 2:
        data = np.mean(data, axis=1) 
    chunk_size = data.shape[0]
    # make time axis
    size = float(chunk_size)  # 波形サイズ
    x = np.arange(0, size/sample_rate, 1.0/sample_rate)

    # 図に描画
    sns.set() # スタイルをきれいにする
    fig = plt.figure(facecolor='w', linewidth=5, edgecolor='black')
    # ax = fig.add_subplot(1, 1, 1, ylim=(-0.5, 0.5)) # 図を1行目1列の1番目に表示(図を1つしか表示しない場合)
    ax = fig.add_subplot(1, 1, 1, title=fig_title, ylim=(ylim_min, ylim_max)) # 図を1行目1列の1番目に表示(図を1つしか表示しない場合)
    ax.set_xlabel('time[s]') # x軸名を設定
    ax.set_ylabel('magnitude') # y軸名を設定
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0)) # x軸の主目盛を1.0ごとに表示
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.10)) # y軸の主目盛を0.10ごとに表示
    file_name = os.path.basename(output_path).split('.')[0] # データの名前を設定
    ax.plot(x, data, label='{}'.format(file_name)) # データをプロット
    # ax.legend(edgecolor="black") # 凡例を追加
    fig.savefig(output_path) # グラフを保存

# スペクトログラムを図にプロットする関数
def spec_plot(base_dir, wav_path, save_path):
    # soxコマンドによりwavファイルからスペクトログラムの画像を生成
    cmd1 = "sox {} -n rate 16.0k spectrogram".format(wav_path)
    subprocess.call(cmd1, shell=True)
    # 生成されたスペクトログラム画像を移動
    #(inference.pyを実行したディレクトリにスペクトログラムが生成されてしまうため)
    spec_path = os.path.join(base_dir, "spectrogram.png")
    cmd2 = "mv {} {}".format(spec_path, save_path)
    subprocess.call(cmd2, shell=True)

# モデルのパラメータ数をカウント
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Juliusを用いた音声認識
def asr_julius(input_file_path):
    # save_path = "~/julius/dictation-kit-4.5/recog_result.txt"
    temp_file = "julius_asr_recog_result.txt"
    # juliusによる音声認識を実行し、結果をファイルに出力
    # 混合ガウスモデル（GMM）ベースの音響モデルを用いる場合→今は「前に進め」、「後ろに退がれ」など（オリジナルの単語辞書に登録されたもの）を認識
    asr_cmd = "echo {} | julius -C ~/julius/dictation-kit-4.5/main.jconf -C ~/julius/dictation-kit-4.5/am-gmm.jconf -nostrip -input rawfile -quiet > {}".format(input_file_path, temp_file)
    # # DNNベースの音響モデルを用いる場合→今はさまざまな日本語を認識（英語は不可）
    # asr_cmd = "echo {} | julius -C ~/julius/dictation-kit-4.5/main.jconf -C ~/julius/dictation-kit-4.5/am-dnn.jconf -dnnconf ~/julius/dictation-kit-4.5/julius.dnnconf -nostrip -input rawfile -quiet > {}".format(input_file_path, save_path)
    subprocess.call(asr_cmd, shell=True)
    # 出力ファイルから認識結果の部分のみを抽出
    with open(temp_file) as f:
        lines = f.readlines()
    recog_text_line = [line.strip() for line in lines if line.startswith('sentence1')] # "sentence1"から始まる行をサーチ
    recog_result = recog_text_line[0][12:-2] # "sentence1: "から"。"の間の文章を抽出
    # 余分なファイルが残らないように削除
    os.remove(temp_file)
    return recog_result


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

    # audio_path = "./test/p232_007/p232_007_mixed.wav"
    # audio_data = load_audio_file(audio_path, audio_length, sample_rate)
    # """audio_data: (num_samples=48000, num_channels=8)"""
    # audio_data = audio_data.transpose(1, 0)
    # """audio_data: (num_channels=8, num_samples=48000)"""
    # multi_complex_spec = wave_to_spec_multi(audio_data, sample_rate, fft_size, hop_length)
    # """multi_complex_spec: (num_channels=8, freq_bins=257, time_frames=301)"""
    # print(multi_complex_spec.shape)
    
    # # マルチチャンネル音声データをスペクトログラムに変換
    # mulichannel_complex_spec = [] # それぞれのチャンネルの複素スペクトログラムを格納するリスト
    # audio_data = load_audio_file(audio_path, audio_length, sample_rate)
    # audio_data = np.asfortranarray(audio_data) # Fortran-contiguousに変換（これがないとエラーが出る）
    # for i in range(audio_channels):
    #     # オーディオデータをスペクトログラムに変換
    #     complex_spec = wave_to_spec(audio_data[:, i], fft_size, hop_length)
    #     mulichannel_complex_spec.append(complex_spec)
    # mulichannel_complex_spec = np.array(mulichannel_complex_spec)
    # """mulichannel_complex_spec: (num_channels=8, freq_bins=257, time_frames=301)"""
    # print(mulichannel_complex_spec.shape)

    # # 残響除去テスト
    # # マスク推定モデルのタイプを指定
    # # model_type = 'Unet' # 'FC' or 'BLSTM' or 'Unet'
    # # # 残響除去手法のタイプを指定
    # # dereverb_type = 'WPE' # None or 'WPE' or 'WPD'
    # # 前処理クラスのインスタンスを作成
    # transform = AudioProcess(sample_rate, fft_size, hop_length)
    # # 目的音のファイルパス
    # target_voice_file = "./test/p232_021_rt0162/p232_021_target.wav"
    # # target_voice_file = "./test/p232_021_rt0162/p232_021_mixed_azimuth60.wav"
    # # 音声データをロード
    # target_audio_data = load_audio_file(target_voice_file, audio_length, sample_rate)
    # """target_audio_data: (num_samples, num_channels)"""
    # # 処理の開始時間
    # start_time = time.perf_counter()
    # # マルチチャンネル音声データを複素スペクトログラムと振幅スペクトログラムに変換（残響除去も実施）
    # estimated_complex_spec, estimated_amp_spec = transform(target_audio_data)
    # """estimated_complex_spec: (num_channels, freq_bins, time_frames), estimated_amp_spec: (num_channels, freq_bins, time_frames)"""
    # estimated_complex_spec = estimated_complex_spec[:, :, :301]
    # # マルチチャンネルスペクトログラムを音声波形に変換
    # multichannel_estimated_voice_data= np.zeros(target_audio_data.shape, dtype='float64') # マルチチャンネル音声波形を格納する配列
    # # 1chごとスペクトログラムを音声波形に変換
    # for i in range(estimated_complex_spec.shape[0]):
    #     # estimated_voice_data = spec_to_wave(estimated_spec[i, :, :], hop_length)
    #     estimated_voice_data = librosa.core.istft(estimated_complex_spec[i, :, :], hop_length=hop_length)
    #     multichannel_estimated_voice_data[:, i] = estimated_voice_data
    # """multichannel_estimated_voice_data: (num_samples, num_channels)"""
    # # 処理の終了時間
    # finish_time = time.perf_counter()
    # print("処理時間：", finish_time - start_time)
    # # オーディオデータを保存
    # estimated_voice_path = os.path.join("./output/wave/", "estimated_voice.wav")
    # save_audio_file(estimated_voice_path, multichannel_estimated_voice_data)
    # # オリジナル音声
    # target_voice_path = os.path.join("./output/wave/", "target_voice.wav")
    # target_voice_data = load_audio_file(target_voice_file, audio_length, sample_rate)
    # save_audio_file(target_voice_path, target_audio_data)
    # # 分離音のスペクトログラム
    # estimated_voice_spec_path = os.path.join("./output/spectrogram/", "estimated_voice.png")
    # spec_plot(os.getcwd(), estimated_voice_path, estimated_voice_spec_path, audio_length)
    # # オリジナル音声のスペクトログラム
    # target_voice_spec_path = os.path.join("./output/spectrogram/", "target_voice.png")
    # spec_plot(os.getcwd(), target_voice_file, target_voice_spec_path, audio_length)

    # Juliusによる音声認識テスト
    ASR_julius("./ref_speech/sample_jp_16kHz.wav")
