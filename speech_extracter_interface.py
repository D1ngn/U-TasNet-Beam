# -*- coding:utf-8 -*-

# 必要モジュールのimport
# pytorch関連
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
# from torch2trt import torch2trt # Xavier上で動かす場合のみ

import os
import numpy as np
import argparse
import time as tm
import soundfile as sf
import pyroomacoustics as pa # 音源定位用
import requests
from io import BytesIO
import socket
import subprocess


# マスクビームフォーマ関連
from models import FCMaskEstimator, BLSTMMaskEstimator, UnetMaskEstimator_kernel3, UnetMaskEstimator_kernel3_single_mask
from beamformer import estimate_covariance_matrix, condition_covariance, estimate_steering_vector, mvdr_beamformer, gev_beamformer, sparse, ds_beamformer, mwf, mvdr_beamformer_two_speakers
from utils.utilities import AudioProcess
# 話者識別用モデル
from utils.embedder import SpeechEmbedder
# 音源分離用モジュール 
from asteroid.models import BaseModel

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

# 音源定位
def localize_music(spec):
    """
    spec: (num_channels, freq_bins, time_frames)
    """
    # MUSIC法を用いて音源定位（cは音速）
    doa = pa.doa.algorithms['MUSIC'](mic_alignments, args.sample_rate, args.fft_size, c=343., num_src=1) # Construct the new DOA object
    doa.locate_sources(spec, freq_range=freq_range)
    speaker_azimuth = doa.azimuth_recon / np.pi * 180.0 # rad→deg
    # 0°〜360°表記を-180°〜180°表記に変更
    if speaker_azimuth[0] > 180:
        speaker_azimuth = int(speaker_azimuth[0] - 360)
    else:
        speaker_azimuth = int(speaker_azimuth[0])
    return speaker_azimuth

# 録音した音声に対する処理を別スレッドで行う関数（録音と音声処理を並列処理）
def speech_extracter(mixed_audio_data):
    # マイクロホンのゲイン調整
    mixed_audio_data = mixed_audio_data * args.mic_gain
    # 雑音除去を行う場合
    if args.denoising_mode:
        # # 処理時間計測
        # start_time = tm.perf_counter()
        # マルチチャンネル音声データを複素スペクトログラムに変換
        mixed_complex_spec = audio_processor.calc_complex_spec(mixed_audio_data)
        """mixed_complex_spec: (num_channels, freq_bins, time_frames)"""
        # 残響除去手法を指定している場合は残響除去処理を実行
        if args.dereverb_type == 'WPE':
            mixed_complex_spec, _ = audio_processor.dereverberation_wpe_multi(mixed_complex_spec)       
        # モデルに入力できるように音声をミニバッチに分けながら振幅スペクトログラムに変換
        mixed_amp_spec_batch = audio_processor.preprocess_mask_estimator(mixed_audio_data, args.chunk_size, device=device)
        """mixed_amp_spec_batch: (batch_size, num_channels, freq_bins, time_frames)"""
        # 発話とそれ以外の雑音の時間周波数マスクを推定
        speech_mask_output, noise_mask_output = model(mixed_amp_spec_batch)
        """speech_mask_output: (batch_size, num_channels, freq_bins, time_frames), noise_mask_output: (batch_size, num_channels, freq_bins, time_frames)"""
        # ミニバッチに分けられたマスクを時間方向に結合し、混合音にかけて各音源のスペクトログラムを取得
        multichannel_speech_spec, _ = audio_processor.postprocess_mask_estimator(mixed_complex_spec, speech_mask_output, args.chunk_size, args.target_aware_channel)
        """multichannel_speech_spec: (num_channels, freq_bins, time_frames)"""
        multichannel_noise_spec, estimated_noise_mask = audio_processor.postprocess_mask_estimator(mixed_complex_spec, noise_mask_output, args.chunk_size, args.noise_aware_channel)
        """multichannel_noise_spec: (num_channels, freq_bins, time_frames)"""
        # 発話のマルチチャンネルスペクトログラムを音声波形に変換
        multichannel_denoised_data = audio_processor.spec_to_wave(multichannel_speech_spec, mixed_audio_data)
        """multichannel_denoised_data: (num_samples, num_channels)"""
        # 1ch分を取り出す
        multichannel_denoised_data = multichannel_denoised_data[:, 0][:, np.newaxis]
        """mixed_speech_data: (num_samples, num_channels=1)"""
        # 話者分離
        separated_audio_data = audio_processor.speaker_separation(speaker_separation_model, multichannel_denoised_data)
        """separated_audio_data: (num_sources, num_samples, num_channels)"""
        # start_time_speeaker_selector = tm.perf_counter()
        # 分離音から目的話者の発話を選出（何番目の発話が目的話者のものかを判断）
        target_speaker_id, speech_amp_spec_all = audio_processor.speaker_selector(embedder, separated_audio_data, ref_dvec, device=device)
        """speech_amp_spec_all: (num_sources, num_channels, freq_bins, time_frames)"""
        # print("ID of the target speaker:", target_speaker_id)
        # finish_time_speeaker_selector = tm.perf_counter()
        # duration_speeaker_selector = finish_time_speeaker_selector - start_time_speeaker_selector
        # rtf = duration_speeaker_selector / (mixed_audio_data.shape[0] / sample_rate)
        # print("実時間比（Speaker Selector）：{:.3f}".format(rtf))
        # 雑音の振幅スペクトログラムを算出
        noise_amp_spec = audio_processor.calc_amp_spec(multichannel_noise_spec)
        """noise_amp_spec: (num_channels, freq_bins, time_frames)"""
        # IRM計算（将来的にはマスクを使わず、信号から直接空間相関行列を算出できるようにする。あるいはcIRMを使う。） TODO
        estimated_target_mask = np.sqrt(speech_amp_spec_all[target_speaker_id] ** 2 / np.maximum((np.sum(speech_amp_spec_all**2, axis=0) + noise_amp_spec ** 2), 1e-7))
        """estimated_target_mask: (num_channels, freq_bins, time_frames)"""
        estimated_interference_mask = np.zeros_like(estimated_target_mask)
        for id in range(speech_amp_spec_all.shape[0]):
            # 目的話者以外の話者の発話マスクを足し合わせる
            if id == target_speaker_id:
                pass
            else:
                estimated_interference_mask += np.sqrt(speech_amp_spec_all[id] ** 2 / np.maximum((np.sum(speech_amp_spec_all**2, axis=0) + noise_amp_spec ** 2), 1e-7))
        """estimated_interference_mask: (num_channels, freq_bins, time_frames)"""
        # 複数チャンネルのうち1チャンネル分のマスクを算出
        if channel_select_type == 'aware':
            # 目的音と干渉音に近いチャンネルのマスクをそれぞれ使用（選択するチャンネルを変えて実験してみるのもあり）
            estimated_target_mask = estimated_target_mask[args.target_aware_channel, :, :]
            estimated_interference_mask = estimated_interference_mask[args.noise_aware_channel, :, :]
        elif channel_select_type == 'median' or channel_select_type == 'single':
            # 複数チャンネル間のマスク値の中央値をとる（median pooling）
            estimated_target_mask = np.median(estimated_target_mask, axis=0)
            estimated_interference_mask = np.median(estimated_interference_mask, axis=0)
        """estimated_target_mask: (freq_bins, time_steps), estimated_interference_mask: (freq_bins, time_frames)"""
        # 目的音のマスクと雑音のマスクからそれぞれの空間共分散行列を推定
        target_covariance_matrix = estimate_covariance_matrix(mixed_complex_spec, estimated_target_mask)
        interference_covariance_matrix = estimate_covariance_matrix(mixed_complex_spec, estimated_interference_mask)
        noise_covariance_matrix = estimate_covariance_matrix(mixed_complex_spec, estimated_noise_mask)
        noise_covariance_matrix = condition_covariance(noise_covariance_matrix, 1e-6) # これがないと性能が大きく落ちる（雑音の共分散行列がスパースであるため？）
        # ビームフォーマによる雑音除去を実行
        if args.beamformer_type == 'MVDR':
            # target_steering_vectors = estimate_steering_vector(target_covariance_matrix)
            # estimated_spec = mvdr_beamformer(mixed_complex_spec, target_steering_vectors, noise_covariance_matrix)
            estimated_target_spec = mvdr_beamformer_two_speakers(mixed_complex_spec, target_covariance_matrix, interference_covariance_matrix, noise_covariance_matrix)
            # estimated_interference_spec = mvdr_beamformer_two_speakers(mixed_complex_spec, interference_covariance_matrix, target_covariance_matrix, noise_covariance_matrix)
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
        # MUSIC法を用いた音源定位
        speaker_azimuth = localize_music(estimated_target_spec)
        print("音源定位結果：", str(speaker_azimuth) + "deg") 
        # マルチチャンネルスペクトログラムを音声波形に変換
        multichannel_estimated_target_voice_data = audio_processor.spec_to_wave(estimated_target_spec, mixed_audio_data)
        # multichannel_estimated_interference_voice_data = audio_processor.spec_to_wave(estimated_interference_spec, mixed_audio_data)
        """multichannel_estimated_target_voice_data: (num_samples, num_channels)"""
        multichannel_estimated_target_voice_data = multichannel_estimated_target_voice_data * 5 # 音量調整のため仮設定 TODO
        # finish_time = tm.perf_counter()
        # print("処理時間：", finish_time - start_time)
        # キューにデータを格納
        # q.put(multichannel_estimated_target_voice_data)
        return multichannel_estimated_target_voice_data, speaker_azimuth
    # 雑音除去を行わない場合
    else:
        if args.channels > 1:
            # 音声波形をスペクトログラムに変換（音源定位用）
            mixed_audio_spec = pa.transform.stft.analysis(mixed_audio_data, L=args.fft_size, hop=args.hop_length)
            """mixed_audio_spec: (time_frames, freq_bins, num_channels)"""
            mixed_audio_spec = mixed_audio_spec.transpose([2, 1, 0])
            """mixed_audio_spec: (num_channels, freq_bins, time_frames)"""
            # MUSIC法を用いた音源定位
            speaker_azimuth = localize_music(mixed_audio_spec)
            print("音源定位結果：", str(speaker_azimuth) + "deg")
        # キューにデータを格納
        # q.put(mixed_audio_data)
        return mixed_audio_data, speaker_azimuth


if __name__ == "__main__":

    # コマンドライン引数を受け取る
    parser = argparse.ArgumentParser(description='Real time voice separation')
    parser.add_argument('-dm', '--denoising_mode', action='store_true', help='whether model denoises audio or not')
    parser.add_argument('-mg', '--mic_gain', type=int, default=1, help='Microphone gain')
    parser.add_argument('-sr', '--sample_rate', type=int, default=16000, help='sampling rate')
    parser.add_argument('-cs', '--chunk_size', type=int, default=48000, help='chunk size of mask estimator and beamformer input')
    parser.add_argument('-c', '--channels', type=int, default=8, help='number of input channels')
    parser.add_argument('-fs', '--fft_size', type=int, default=512, help='size of fast fourier transform')
    parser.add_argument('-hl', '--hop_length', type=int, default=160, help='number of audio samples between adjacent STFT columns')
    parser.add_argument('-twd', '--temp_wav_dir', type=str, default="./temp/", help='directory for storaging temporaly wave file')
    parser.add_argument('-mt', '--model_type', type=str, default='Unet_single_mask', help='type of mask estimator model')
    parser.add_argument('-sst', '--speaker_separator_type', type=str, default='conv_tasnet', help='type of speaker separator model')
    parser.add_argument('-bt', '--beamformer_type', type=str, default='MVDR', help='type of beamformer (DS or MVDR or GEV or MWF)')
    parser.add_argument('-dt', '--dereverb_type', type=str, default='None', help='type of dereverb algorithm (None or WPE)')
    parser.add_argument('-ep', '--embedder_path', type=str, default="./utils/embedder.pt", help='path of pretrained embedder model')
    parser.add_argument('-rsp', '--ref_speech_path', type=str, default="./utils/ref_speech/sample.wav", help='path of reference speech')
    parser.add_argument('-tac', '--target_aware_channel', type=int, default=0, help='microphone channel near target source')
    parser.add_argument('-nac', '--noise_aware_channel', type=int, default=4, help='microphone channel near noise source')
    args = parser.parse_args()

    #########################音源定位用設定########################
    freq_range = [200, 3000] # 空間スペクトルの算出に用いる周波数帯[Hz]
    # TAMAGO-03マイクロホンアレイにおける各マイクロホンの空間的な位置関係
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
    """mic_alignments: (num_microphones, 3D coordinates [m])"""
    # 各マイクロホンの空間的な位置関係を表す配列
    mic_alignments = mic_alignments.T # get the microphone arra
    """mic_alignments: (3D coordinates [m], num_microphones)"""
    #############################################################

    #################################通信用設定#################################
    # Nakbotクライアントからの接続用
    ss_host = "127.0.0.1"
    ss_port = 1234
    # JuliusサーバのURL
    # url = "http://192.168.10.101:8000/asr_julius"
    url = "http://192.168.10.116:8000/asr_julius" # Jetson Xavierではこっちを使用
    ses = requests.Session()
    ###########################################################################

    # GPUが使える場合はGPUを使用、使えない場合はCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：" , device)
    # ネットワークモデルと学習済みのパラメータを保存したチェックポイントファイルのパスを指定
    if args.model_type == 'BLSTM':
        model = BLSTMMaskEstimator()
        channel_select_type = 'median'
        padding = False
        checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_BLSTM_1201/ckpt_epoch70.pt" # BLSTM small
    elif args.model_type == 'FC':
        model = FCMaskEstimator()
        channel_select_type = 'aware'
        padding = False
        checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_all_FC_1202/ckpt_epoch80.pt" # FC small
    elif args.model_type == 'Unet':
        channel_select_type = 'aware'
        padding = True
        model = UnetMaskEstimator_kernel3()
        checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_Unet_aware_1208/ckpt_epoch110.pt" # U-Net aware channel
    elif args.model_type == 'Unet_single_mask':
        model = UnetMaskEstimator_kernel3_single_mask()
        channel_select_type = 'single'
        padding = True
        checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_multi_wav_test_original_length_Unet_single_mask_median_multisteplr00001start_20210701/ckpt_epoch190.pt" # U-Net-single-mask
    else:
        pass

    # 前処理クラスのインスタンスを作成
    audio_processor = AudioProcess(args.sample_rate, args.fft_size, args.hop_length, channel_select_type, padding)
    # ネットワークをCPUまたはGPUへ
    model.to(device)
    # 学習済みのパラメータをロード
    model_params = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_params['model_state_dict'])
    # Unetを使って推論
    # ネットワークを推論モードへ
    model.eval()
    # # 入力サンプルとともにTensorRTに変換
    # tmp = torch.ones((1, args.channels, int(args.fft_size/2)+1, 513)).to(device)
    # model = torch2trt(model, [tmp])
    # model = torch2trt(model, [tmp], fp16_mode=True) # 精度によってモード切り替え
    # 話者分離モデルを指定
    if args.speaker_separator_type == 'conv_tasnet':
        # pretrained_param_speaker_separation = "JorisCos/ConvTasNet_Libri2Mix_sepclean_16k" # ConvTasNet 16kHz
        pretrained_param_speaker_separation = "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k" # ConvTasNet 16kHz noisy ← こっちの方が精度が高そう
        # 話者分離モデルの学習済みパラメータをダウンロード
        speaker_separation_model = BaseModel.from_pretrained(pretrained_param_speaker_separation)
    # elif args.speaker_separator_type == 'sepformer':
    speaker_separation_model.to(device)
    # 話者識別モデルの学習済みパタメータをロード（いずれはhparamsでパラメータを指定できる様にする TODO）
    embedder = SpeechEmbedder()
    embedder.to(device)
    embed_params = torch.load(args.embedder_path, map_location=device)
    embedder.load_state_dict(embed_params)
    embedder.eval()
    # 声を分離抽出したい人の発話サンプルをロードし、評価用に保存
    ref_speech_data, _ = sf.read(args.ref_speech_path)
    # 発話サンプルの特徴量（ログメルスペクトログラム）をベクトルに変換
    if ref_speech_data.ndim == 1:
        ref_speech_data = ref_speech_data[:, np.newaxis]
    ref_complex_spec = audio_processor.calc_complex_spec(ref_speech_data)
    ref_log_mel_spec = audio_processor.calc_log_mel_spec(ref_complex_spec)
    ref_log_mel_spec = torch.from_numpy(ref_log_mel_spec).float().to(device)
    ref_dvec = embedder(ref_log_mel_spec[0]) # 入力は1ch分
    """ref_dvec: (embed_dim=256,)"""

    # Nakbotクライアント（駆動側）から送られてきた音声データを受け取って処理
    # AF_INETはIPv4、SOCK_STREAMはTCPであることを表す
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # IPとPORTを指定してバインド
    server_sock.bind((ss_host, ss_port))

    # 接続を待機
    server_sock.listen()
    print("Wainting for connections...")
    # 接続された場合接続先の情報を格納
    client_sock, client_address = server_sock.accept()
    # 一時的にデータを保存するための音声ファイルパス
    os.makedirs(args.temp_wav_dir, exist_ok=True)
    temp_wav_path = os.path.join(args.temp_wav_dir, "temp.wav")
    while True:
        # データ受け取り（引数でデータサイズを指定）
        recv_msg = client_sock.recv(1024)
        # バイナリデータを文字列に変換
        raw_path = recv_msg.decode('utf-8')
        # データが空の場合は以降の処理をスキップ
        if len(raw_path) == 0:
            continue
        # soxコマンドによりrawファイルをwavファイルに変換
        raw2wav_cmd = "sox -t raw -r 16k -e signed -b 16 -c 8 -x {} {}".format(raw_path, temp_wav_path)
        subprocess.call(raw2wav_cmd, shell=True)
        # rawファイルを削除する
        os.remove(raw_path)
        # wavファイルをnumpy配列として読み込み
        temp_wav_data, _ = sf.read(temp_wav_path)
        # 目的話者の発話抽出と位置検出を実行
        temp_audio_data, speaker_azimuth = speech_extracter(temp_wav_data)
        # 1ch分を取り出し、バイナリデータに変換
        if temp_audio_data.ndim == 2:
            temp_audio_data = temp_audio_data[:, 0]
        out = BytesIO()
        np.save(out, temp_audio_data)
        binary = out.getvalue()
        # 音声認識サーバに送信し、認識結果を受信
        result = ses.post(url, files={'myFile': binary})
        print("音声認識結果：", result.text)
        # 音声認識結果と音源定位結果を結合（"\0"はC++で文字列配列を扱う際に必要）
        send_data = result.text.strip() + "|" + str(speaker_azimuth) + "\0"
        # データをバイナリデータに変換してクライアントに送信
        client_sock.send(send_data.encode('utf-8'))
