#!/usr/bin/env python
# coding: utf-8

# torch関連のモジュールのimport
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
# from torch2trt import torch2trt # Xavier上で動かす場合のみ
# その他必要モジュールのimport
import os
import glob
import numpy as np
import argparse
import time
import soundfile as sf

# WARNINGの表示を全て消す場合
import warnings
warnings.simplefilter('ignore')

from models import FCMaskEstimator, BLSTMMaskEstimator, UnetMaskEstimator_kernel3, CNNMaskEstimator_kernel3, UnetMaskEstimator_kernel3_single_mask, UnetMaskEstimator_kernel3_single_mask_two_speakers # 雑音・残響除去モデル各種
from beamformer import estimate_covariance_matrix, condition_covariance, estimate_steering_vector, mvdr_beamformer, mvdr_beamformer_two_speakers, gev_beamformer, sparse, ds_beamformer, mwf, localize_music # ビームフォーマ各種
from utils.utilities import AudioProcess, spec_plot, count_parameters, wave_plot # 音声処理用
from utils.embedder import SpeechEmbedder # 話者識別用
from utils.evaluate import audio_eval, asr_eval # 評価用
from utils.asr import ASR # 音声認識用
from utils.asr import asr_julius # 音声認識用
from asteroid.models import BaseModel # 音源分離用


def main():
    # コマンドライン引数を受け取る
    parser = argparse.ArgumentParser(description='Real time voice separation')
    parser.add_argument('-sr', '--sample_rate', type=int, default=16000, help='sampling rate') # サンプリング周波数
    parser.add_argument('-bl', '--batch_length', type=int, default=48000, help='batch size of mask estimator and beamformer input') # 音声をバッチ処理する際の1バッチ当たりのサンプル数
    parser.add_argument('-c', '--channels', type=int, default=8, help='number of input channels') # マイクのチャンネル数
    parser.add_argument('-fs', '--fft_size', type=int, default=512, help='size of fast fourier transform') # 高速フーリエ変換のフレームサイズ
    parser.add_argument('-hl', '--hop_length', type=int, default=160, help='number of audio samples between adjacent STFT columns') # 高速フーリエ変換におけるフレームのスライド幅
    parser.add_argument('-mt', '--model_type', type=str, default='Unet_single_mask', help='type of mask estimator model (FC or BLSTM or CNN or Unet or Unet_single_mask or Unet_single_mask_two_speakers)') # 雑音（残響除去）モデルのタイプ
    parser.add_argument('-sst', '--speaker_separator_type', type=str, default='conv_tasnet', help='type of speaker separator model (conv_tasnet)') # 話者分離モデルのタイプ
    parser.add_argument('-bt', '--beamformer_type', type=str, default='MVDR', help='type of beamformer (DS or MVDR or GEV or MWF)') # ビームフォーマのタイプ
    parser.add_argument('-dt', '--dereverb_type', type=str, default='None', help='type of dereverb algorithm (None or WPE)') # 残響除去手法のタイプ
    parser.add_argument('-ep', '--embedder_path', type=str, default="./utils/embedder.pt", help='path of pretrained embedder model') # 話者識別用の学習済みモデルのパス
    parser.add_argument('-rsp', '--ref_speech_path', type=str, default="./utils/ref_speech/sample.wav", help='path of reference speech') # 声を抽出したい人の発話サンプルのパス
    parser.add_argument('-tac', '--target_aware_channel', type=int, default=0, help='microphone channel near target source') # 目的音に関するマスクのチャンネル
    parser.add_argument('-nac', '--noise_aware_channel', type=int, default=4, help='microphone channel near noise source') # 雑音に関するマスクのチャンネル
    args = parser.parse_args()

    # # 処理後の音声の振幅の最大値を処理前の混合音声の振幅の最大値に合わせる（True）か否か（False）
    # fit_max_value = False

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
    
    # 英語（男性）
    # 3秒版
    # target_voice_file = "./test/p232_016/p232_016_target.wav"
    # interference_audio_file = "./test/p232_016/p232_016_interference_azimuth45.wav"
    # mixed_audio_file = "./test/p232_016/p232_016_mixed_azimuth45.wav"
    # target_voice_file = "./test/p232_021/p232_021_target.wav"
    # interference_audio_file = "./test/p232_021/p232_021_interference_azimuth15.wav"
    # mixed_audio_file = "./test/p232_021/p232_021_mixed_azimuth15.wav"
    # オリジナルの長さ版
    # target_voice_file = "./test/p232_123/p232_123_target.wav" 
    # interference_audio_file = "./test/p232_123/p232_123_interference_azimuth15.wav"
    # mixed_audio_file = "./test/p232_123/p232_123_mixed_azimuth15.wav" # 違いがわからない
    # 英語（女性）
    # 3秒版
    # target_voice_file = "./test/p257_006/p257_006_target.wav"
    # interference_audio_file = "./test/p257_006/p257_006_interference_azimuth60.wav"
    # mixed_audio_file = "./test/p257_006/p257_006_mixed_azimuth60.wav"
    # target_voice_file = "./test/p257_130/p257_130_target.wav"
    # interference_audio_file = "./test/p257_130/p257_130_interference_azimuth0.wav"
    # mixed_audio_file = "./test/p257_130/p257_130_mixed_azimuth0.wav"
    # オリジナルの長さ版
    # target_voice_file = "./test/p257_011/p257_011_target.wav"
    # interference_audio_file = "./test/p257_011/p257_011_interference_azimuth15.wav"
    # mixed_audio_file = "./test/p257_011/p257_011_mixed_azimuth15.wav" # 結構いい結果が出る
    # target_voice_file = "./test/p257_050/p257_050_target.wav"
    # interference_audio_file = "./test/p257_050/p257_050_interference_azimuth15.wav"
    # mixed_audio_file = "./test/p257_050/p257_050_mixed_azimuth15.wav"
    # target_voice_file = "./test/p257_430/p257_430_target.wav"
    # interference_audio_file = "./test/p257_430/p257_430_interference_azimuth15.wav"
    # mixed_audio_file = "./test/p257_430/p257_430_mixed_azimuth15.wav" # めちゃくちゃわかりやすい結果が出る
    # オリジナルの長さ版（残響あり）
    # target_voice_file = "./test/p257_430_rt0161/p257_430_target.wav"
    # interference_audio_file = "./test/p257_430_rt0161/p257_430_interference_azimuth15.wav"
    # mixed_audio_file = "./test/p257_430_rt0161/p257_430_mixed_azimuth15.wav" # なぜか残響なし版と比べて混合音声のWERが改善？
    # 日本語（女性）
    # target_voice_file = "./test/JVS/BASIC5000_0145_target.wav"
    # interference_audio_file = "./test/JVS/BASIC5000_0145_interference.wav"
    # mixed_audio_file = "./test/JVS/BASIC5000_0145_mixed.wav"
    # 英語（男性・残響あり）
    # target_voice_file = "./test/p232_021_rt0162/p232_021_target.wav"
    # interference_audio_file = "./test/p232_021_rt0162/p232_021_interference_azimuth60.wav"
    # mixed_audio_file = "./test/p232_021_rt0162/p232_021_mixed_azimuth60.wav"
    # 英語話者2人（男性1人＋女性1人）版
    target_voice_file = "./test/p232_414_p257_074_noise_mix/p232_414_target.wav"
    interference_audio_file = "./test/p232_414_p257_074_noise_mix/p232_414_p257_074_interference_azimuth15.wav"
    noise_file = "./test/p232_414_p257_074_noise_mix/p232_414_p257_074_noise_azimuth180.wav"
    mixed_audio_file = "./test/p232_414_p257_074_noise_mix/p232_414_p257_074_mixed.wav"
    # target_voice_file = "./test/p232_231_p257_129_noise_mix/p232_231_target.wav"
    # interference_audio_file = "./test/p232_231_p257_129_noise_mix/p232_231_p257_129_interference_azimuth15.wav"
    # noise_file = "./test/p232_231_p257_129_noise_mix/p232_231_p257_129_noise_azimuth180.wav"
    # mixed_audio_file = "./test/p232_231_p257_129_noise_mix/p232_231_p257_129_mixed.wav" # こっちの方が結果がわかりやすい
    # 日本語話者2人版
    # # 男性1人＋女性1人
    # target_voice_file = "./test/BASIC5000_0001_BASIC5000_0034_mix/BASIC5000_0001.wav"
    # interference_audio_file = "./test/BASIC5000_0001_BASIC5000_0034_mix/BASIC5000_0034.wav"
    # mixed_audio_file = "./test/BASIC5000_0001_BASIC5000_0034_mix/BASIC5000_0001_BASIC5000_034_mixed.wav"

    wave_dir = "./output/wave/"
    os.makedirs(wave_dir, exist_ok=True)
    # オーディオファイルに対応する音声の波形を保存
    wave_image_dir = "./output/wave_image/"
    os.makedirs(wave_image_dir, exist_ok=True)
    # オーディオファイルに対応するスペクトログラムを保存
    spec_dir = "./output/spectrogram/"
    os.makedirs(spec_dir, exist_ok=True)

    # 音声認識精度評価用正解ラベルを格納したディレクトリを指定
    reference_label_dir = "../AudioDatasets/NoisySpeechDatabase/testset_txt/"

    # 学習済みのパラメータを保存したチェックポイントファイルのパスを指定
    # NoisySpeechDataset_for_unet_fft_512_multi_wav_1207で学習
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_all_BLSTM_1202/ckpt_epoch80.pt" # BLSTM
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_all_FC_1202/ckpt_epoch80.pt" # FC
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_BLSTM_1201/ckpt_epoch70.pt" # BLSTM small
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_BLSTM_1211/ckpt_epoch80.pt" # BLSTM small2
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_Unet_aware_1208/ckpt_epoch110.pt" # U-Net aware channel←元ベストモデル
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_Unet_aware_1211/ckpt_epoch160.pt" # U-Net aware channel2
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_Unet_median_1209/ckpt_epoch50.pt" # U-Net median operation
    # NoisySpeechDataset_for_unet_fft_512_multi_wav_1209で学習
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_logmel_BLSTM_1209/ckpt_epoch100.pt" # BLSTM-logmel
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_BLSTM2_1231/ckpt_epoch100.pt" # BLSTM2 small
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_Unet_aware_20210111/ckpt_epoch100.pt" # U-Net aware channel←ベストモデル
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_CNN_aware_20210310/ckpt_epoch200.pt"
    # checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_for_unet_fft_512_multi_wav_Unet_single_mask_median_20210315/ckpt_epoch170.pt" # U-Net-single-mask small data  (best model)
    checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_multi_wav_test_original_length_Unet_single_mask_median_multisteplr00001start_20210701/ckpt_epoch190.pt" # U-Net-single-mask small data (newest model)

    # 音声認識結果を保存するディレクトリを指定
    recog_result_dir = "./recog_result/{}_{}_{}_dereverb_type_{}/".format(target_voice_file.split('/')[-2], args.model_type, args.beamformer_type, str(args.dereverb_type))
    os.makedirs(recog_result_dir, exist_ok=True)

    # GPUが使える場合はGPUを使用、使えない場合はCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：" , device)

    # ネットワークモデルの定義、チャンネルの選び方の指定、モデル入力時にパディングを行うか否かを指定
    if args.model_type == 'BLSTM':
        model = BLSTMMaskEstimator()
        channel_select_type = 'median'
        padding = False
    elif args.model_type == 'FC':
        model = FCMaskEstimator()
        channel_select_type = 'aware'
        padding = False
    elif args.model_type == 'CNN':
        model = CNNMaskEstimator_kernel3()
        channel_select_type = 'aware'
        padding = True
    elif args.model_type == 'Unet':
        model = UnetMaskEstimator_kernel3()
        channel_select_type = 'aware'
        padding = True
    elif args.model_type == 'Unet_single_mask':
        model = UnetMaskEstimator_kernel3_single_mask()
        channel_select_type = 'single'
        padding = True
    elif args.model_type == 'Unet_single_mask_two_speakers':
        model = UnetMaskEstimator_kernel3_single_mask_two_speakers()
        channel_select_type = 'single'
        padding = True

    # 音声処理クラスのインスタンスを作成
    audio_processor = AudioProcess(args.sample_rate, args.fft_size, args.hop_length, channel_select_type, padding)

    # ネットワークをCPUまたはGPUへ
    model.to(device)
    # 学習済みのパラメータをロード
    model_params = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_params['model_state_dict'])
    # print("モデルのパラメータ数：", count_parameters(model))
    # ネットワークを推論モードへ
    model.eval()
    # # 入力サンプルとともにTensorRTに変換
    # tmp = torch.ones((1, args.channels, int(args.fft_size/2)+1, 513)).to(device)
    # # model = torch2trt(model, [tmp])
    # model = torch2trt(model, [tmp], fp16_mode=True) # 精度によってモード切り替え
    # 音声認識用のインスタンスを生成
    asr_ins = ASR(lang='eng')
    # 話者分離モデルの学習済みパラメータをダウンロード
    if args.speaker_separator_type == 'conv_tasnet':
        # 「https://huggingface.co/models?filter=asteroid」にある話者分離用の学習済みモデルを指定
        # pretrained_param_speaker_separation = "JorisCos/ConvTasNet_Libri2Mix_sepclean_16k" # ConvTasNet 16kHz
        pretrained_param_speaker_separation = "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k" # ConvTasNet 16kHz noisy ← こっちの方が精度が高そう
        speaker_separation_model = BaseModel.from_pretrained(pretrained_param_speaker_separation)
    else:
        print("Please specify the correct speaker separator type")
    speaker_separation_model.to(device)
    # tmp = torch.ones(args.batch_length, 1)
    # speaker_separation_model = torch2trt(speaker_separation_model, [tmp])
    # 話者識別モデルの学習済みパタメータをロード（いずれはhparamsでパラメータを指定できる様にする TODO）
    embedder = SpeechEmbedder()
    embedder.to(device)
    embed_params = torch.load(args.embedder_path, map_location=device)
    embedder.load_state_dict(embed_params)
    embedder.eval()
    # 声を分離抽出したい人の発話サンプルをロードし、評価用に保存
    ref_speech_data, _ = sf.read(args.ref_speech_path)
    # シングルチャンネル音声の場合はチャンネルの次元を追加
    if ref_speech_data.ndim == 1:
        ref_speech_data = ref_speech_data[:, np.newaxis]
    ref_speech_save_path = os.path.join(wave_dir, "reference_voice.wav")
    sf.write(ref_speech_save_path, ref_speech_data, args.sample_rate)
    # 発話サンプルの特徴量（ログメルスペクトログラム）をベクトルに変換
    ref_complex_spec = audio_processor.calc_complex_spec(ref_speech_data)
    ref_log_mel_spec = audio_processor.calc_log_mel_spec(ref_complex_spec)
    ref_log_mel_spec = torch.from_numpy(ref_log_mel_spec).float().to(device)
    # 入力サンプルとともにTensorRTに変換
    # embedder = torch2trt(embedder, [torch.unsqueeze(ref_log_mel_spec[0], 0)])
    ref_dvec = embedder(ref_log_mel_spec[0]) # 入力は1ch分
    """ref_dvec: (embed_dim=256,)"""

    # 処理の開始時間
    start_time = time.perf_counter()
    # 音声データをロード
    mixed_audio_data, _ = sf.read(mixed_audio_file)
    """mixed_audio_data: (num_samples, num_channels)"""
    # 元のデータの振幅の最大値を記録
    max_amp_preprocess = mixed_audio_data.max()
    # マルチチャンネル音声データを複素スペクトログラムに変換
    mixed_complex_spec = audio_processor.calc_complex_spec(mixed_audio_data)
    """mixed_complex_spec: (num_channels, freq_bins, time_frames)"""

    # 残響除去手法を指定している場合は残響除去処理を実行
    if args.dereverb_type == 'WPE':
        mixed_complex_spec, _ = audio_processor.dereverberation_wpe_multi(mixed_complex_spec)
        
    # モデルに入力できるように音声をミニバッチに分けながら振幅スペクトログラムに変換
    mixed_amp_spec_batch = audio_processor.preprocess_mask_estimator(mixed_audio_data, args.batch_length, device=device)
    """mixed_amp_spec_batch: (batch_size, num_channels, freq_bins, time_frames)"""
    # 発話とそれ以外の雑音の時間周波数マスクを推定
    speech_mask_output, noise_mask_output = model(mixed_amp_spec_batch)
    """speech_mask_output: (batch_size, num_channels, freq_bins, time_frames), noise_mask_output: (batch_size, num_channels, freq_bins, time_frames)"""
    # ミニバッチに分けられたマスクを時間方向に結合し、混合音にかけて各音源のスペクトログラムを取得
    multichannel_speech_spec, _ = audio_processor.postprocess_mask_estimator(mixed_complex_spec, speech_mask_output, args.batch_length, args.target_aware_channel)
    """multichannel_speech_spec: (num_channels, freq_bins, time_frames)"""
    multichannel_noise_spec, estimated_noise_mask = audio_processor.postprocess_mask_estimator(mixed_complex_spec, noise_mask_output, args.batch_length, args.noise_aware_channel)
    """multichannel_noise_spec: (num_channels, freq_bins, time_frames)"""
    # 発話のマルチチャンネルスペクトログラムを音声波形に変換
    multichannel_denoised_data = audio_processor.spec_to_wave(multichannel_speech_spec, mixed_audio_data)
    """multichannel_denoised_data: (num_samples, num_channels)"""

    # 1ch分を取り出す
    multichannel_denoised_data = multichannel_denoised_data[:, 0][:, np.newaxis]
    """multichannel_denoised_data: (num_samples, num_channels=1)"""

    # 話者分離
    separated_audio_data = audio_processor.speaker_separation(speaker_separation_model, multichannel_denoised_data)
    """separated_audio_data: (num_sources, num_samples, num_channels)"""
    # start_time_speeaker_selector = time.perf_counter()
    # 分離音から目的話者の発話を選出（何番目の発話が目的話者のものかを判断）
    target_speaker_id, speech_amp_spec_all = audio_processor.speaker_selector(embedder, separated_audio_data, ref_dvec, device=device)
    """speech_amp_spec_all: (num_sources, num_channels, freq_bins, time_frames)"""
    # print("ID of the target speaker:", target_speaker_id)
    # finish_time_speeaker_selector = time.perf_counter()
    # duration_speeaker_selector = finish_time_speeaker_selector - start_time_speeaker_selector
    # rtf = duration_speeaker_selector / (mixed_audio_data.shape[0] / args.sample_rate)
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
    # noise_covariance_matrix /= np.trace(noise_covariance_matrix, axis1=-2, axis2=-1)[..., None, None]
    # ビームフォーマによる雑音除去を実行
    if args.beamformer_type == 'MVDR':
        # target_steering_vectors = estimate_steering_vector(target_covariance_matrix)
        # estimated_spec = mvdr_beamformer(mixed_complex_spec, target_steering_vectors, noise_covariance_matrix)
        estimated_target_spec = mvdr_beamformer_two_speakers(mixed_complex_spec, target_covariance_matrix, interference_covariance_matrix, noise_covariance_matrix)
        # estimated_interference_spec = mvdr_beamformer_two_speakers(mixed_complex_spec, interference_covariance_matrix, target_covariance_matrix, noise_covariance_matrix)
    elif args.beamformer_type == 'GEV':
        estimated_target_spec = gev_beamformer(mixed_complex_spec, target_covariance_matrix, noise_covariance_matrix)
    elif args.beamformer_type == "DS":
        target_steering_vectors = estimate_steering_vector(target_covariance_matrix)
        estimated_target_spec = ds_beamformer(mixed_complex_spec, target_steering_vectors)
    elif args.beamformer_type == "MWF":
        estimated_target_spec = mwf(mixed_complex_spec, target_covariance_matrix, noise_covariance_matrix)
    elif args.beamformer_type == 'Sparse':
        estimated_target_spec = sparse(mixed_complex_spec, estimated_target_mask) # マスクが正常に推定できているかどうかをテストする用
    else:
        print("Please specify the correct beamformer type")
    """estimated_target_spec: (num_channels, freq_bins, time_frames)"""
    # マルチチャンネルスペクトログラムを音声波形に変換
    multichannel_estimated_target_voice_data = audio_processor.spec_to_wave(estimated_target_spec, mixed_audio_data)
    # multichannel_estimated_interference_voice_data = audio_processor.spec_to_wave(estimated_interference_spec, mixed_audio_data)
    """multichannel_estimated_target_voice_data: (num_samples, num_channels)"""

    # # 最大値を元の音声に合わせる場合
    # if fit_max_value:
    #     max_amp_postprocess = multichannel_estimated_target_voice_data.max()
    #     multichannel_estimated_target_voice_data *= max_amp_preprocess / max_amp_postprocess

    # 処理の終了時間
    finish_time = time.perf_counter()
    # 処理時間
    process_time = finish_time - start_time
    print("処理時間：{:.3f}sec".format(process_time))
    # 実時間比（Real Time Factor）
    rtf = process_time / (mixed_audio_data.shape[0] / args.sample_rate)
    print("実時間比：{:.3f}".format(rtf))

    # MUSIC法を用いた音源定位
    speaker_azimuth = localize_music(estimated_target_spec, mic_alignments, args.sample_rate, args.fft_size)
    print("音源定位結果：", str(speaker_azimuth) + "deg") 

    # オーディオデータを保存
    estimated_target_voice_path = os.path.join(wave_dir, "estimated_target_voice.wav")
    sf.write(estimated_target_voice_path, multichannel_estimated_target_voice_data, args.sample_rate)
    # estimated_interference_voice_path = os.path.join(wave_dir, "estimated_interference_voice.wav")
    # sf.write(estimated_interference_voice_path, multichannel_estimated_interference_voice_data, args.sample_rate)
    # 雑音除去後の混合音を保存
    denoised_voice_path = os.path.join(wave_dir, "denoised_voice.wav")
    sf.write(denoised_voice_path, multichannel_denoised_data, args.sample_rate)
    # デバッグ用に元のオーディオデータとそのスペクトログラムを保存
    # 目的話者の発話
    target_voice_path = os.path.join(wave_dir, "target_voice.wav")
    target_voice_data, _ = sf.read(target_voice_file)
    sf.write(target_voice_path, target_voice_data, args.sample_rate)
    # 干渉話者の発話
    interference_audio_path = os.path.join(wave_dir, "interference_audio.wav")
    interference_audio_data, _ = sf.read(interference_audio_file)
    sf.write(interference_audio_path, interference_audio_data, args.sample_rate)
    # 雑音
    noise_path = os.path.join(wave_dir, "noise.wav")
    noise_data, _ = sf.read(noise_file)
    sf.write(noise_path, noise_data, args.sample_rate)
    # 混合音声
    mixed_audio_path = os.path.join(wave_dir, "mixed_audio.wav")
    sf.write(mixed_audio_path, mixed_audio_data, args.sample_rate)

    # 音声の波形を画像として保存（マルチチャンネル未対応）
    # 目的話者の発話の波形
    target_voice_img_path = os.path.join(wave_image_dir, "target_voice.png")
    wave_plot(target_voice_path, target_voice_img_path, ylim_min=-1.0, ylim_max=1.0)
    # 干渉話者の発話の波形
    interference_img_path = os.path.join(wave_image_dir, "interference_audio.png")
    wave_plot(interference_audio_path, interference_img_path, ylim_min=-1.0, ylim_max=1.0)
    # 雑音
    noise_img_path = os.path.join(wave_image_dir, "noise.png")
    wave_plot(noise_path, noise_img_path, ylim_min=-1.0, ylim_max=1.0)
    # 分離音の波形
    estimated_voice_img_path = os.path.join(wave_image_dir, "estimated_target_voice.png")
    wave_plot(estimated_target_voice_path, estimated_voice_img_path, ylim_min=-1.0, ylim_max=1.0)
    # 目的話者の発話サンプルの波形
    ref_speech_img_path = os.path.join(wave_image_dir, "ref_speech.png")
    wave_plot(args.ref_speech_path, ref_speech_img_path, ylim_min=-1.0, ylim_max=1.0)
    # 混合音声の波形
    mixed_audio_img_path = os.path.join(wave_image_dir, "mixed_audio.png")
    wave_plot(mixed_audio_path, mixed_audio_img_path, ylim_min=-1.0, ylim_max=1.0)

    # スペクトログラムを画像として保存
    # 現在のディレクトリ位置を取得
    base_dir = os.getcwd()
    # オリジナル音声のスペクトログラム
    target_voice_spec_path = os.path.join(spec_dir, "target_voice.png")
    spec_plot(base_dir, target_voice_path, target_voice_spec_path)
    # 外的雑音のスペクトログラム
    interference_audio_spec_path = os.path.join(spec_dir, "interference_audio.png")
    spec_plot(base_dir, interference_audio_path, interference_audio_spec_path)
    # 分離音のスペクトログラム
    estimated_voice_spec_path = os.path.join(spec_dir, "estimated_target_voice.png")
    spec_plot(base_dir, estimated_target_voice_path, estimated_voice_spec_path)
    # 混合音声のスペクトログラム
    mixed_audio_spec_path = os.path.join(spec_dir, "mixed_audio.png")
    spec_plot(base_dir, mixed_audio_path, mixed_audio_spec_path)

    # 音源分離性能の評価
    sdr_mix, sir_mix, sar_mix, sdr_est, sir_est, sar_est = \
        audio_eval(args.sample_rate, target_voice_path, interference_audio_path, mixed_audio_path, estimated_target_voice_path)
    
    # 音声認識性能の評価
    # ESPnetを用いる場合
    target_voice_recog_text = asr_ins.speech_recognition(target_voice_path) # （例） IT IS MARVELLOUS
    target_voice_recog_text = target_voice_recog_text.replace('.', '').upper().split() # （例） ['IT', 'IS', 'MARVELLOUS']
    mixed_audio_recog_text = asr_ins.speech_recognition(mixed_audio_path)
    mixed_audio_recog_text = mixed_audio_recog_text.replace('.', '').upper().split()
    estimated_voice_recog_text = asr_ins.speech_recognition(estimated_target_voice_path)
    estimated_voice_recog_text = estimated_voice_recog_text.replace('.', '').upper().split()
    # ファイル名を取得
    file_num = os.path.basename(target_voice_file).split('.')[0].rsplit('_', maxsplit=1)[0] # （例） p232_016
    # 正解ラベルを読み込む
    reference_label_path = os.path.join(reference_label_dir, file_num + '.txt')
    with open(reference_label_path, 'r', encoding="utf8") as ref:
        # ピリオドとコンマを消して大文字に変換した後、スペースで分割
        reference_label_text = ref.read().replace('.', '').replace(',', '').upper().split()
    # WERを計算
    clean_recog_result_save_path = os.path.join(recog_result_dir, file_num + '_clean.txt')
    mix_recog_result_save_path = os.path.join(recog_result_dir, file_num + '_mix.txt')
    est_recog_result_save_path = os.path.join(recog_result_dir, file_num + '_est.txt')
    wer_clean = asr_eval(reference_label_text, target_voice_recog_text, clean_recog_result_save_path)
    wer_mix = asr_eval(reference_label_text, mixed_audio_recog_text, mix_recog_result_save_path)
    wer_est = asr_eval(reference_label_text, estimated_voice_recog_text, est_recog_result_save_path)


    # # Juliusを用いる場合（日本語シングルチャンネル音声のみに対応）
    # # 目的音
    # target_voice_data, _ = sf.read(target_voice_path)
    # # マルチチャンネル音声の場合は1ch目を取り出す
    # if target_voice_data.ndim == 2:
    #     target_voice_1ch_path = "./utils/target_voice_1ch.wav"
    #     sf.write(target_voice_1ch_path, target_voice_data[:, 0], args.sample_rate)
    #     target_voice_recog_text = asr_julius(target_voice_1ch_path) # （例） IT IS MARVELLOUS
    #     os.remove(target_voice_1ch_path)
    # else:    
    #     target_voice_recog_text = asr_julius(target_voice_path) # （例） IT IS MARVELLOUS
    # target_voice_recog_text = target_voice_recog_text.split() # （例） ['IT', 'IS', 'MARVELLOUS']
    # # 混合音
    # mixed_audio_data, _ = sf.read(mixed_audio_path)
    # if mixed_audio_data.ndim == 2:
    #     mixed_audio_1ch_path = "./utils/mixed_audio_1ch.wav"
    #     sf.write(mixed_audio_1ch_path, mixed_audio_data[:, 0], args.sample_rate)
    #     mixed_audio_recog_text = asr_julius(mixed_audio_1ch_path)
    #     os.remove(mixed_audio_1ch_path)
    # else:    
    #     mixed_audio_recog_text = asr_julius(mixed_audio_path)
    # mixed_audio_recog_text = mixed_audio_recog_text.split()
    # # 処理後の目的音
    # estimated_target_voice_data, _ = sf.read(estimated_target_voice_path)
    # if estimated_target_voice_data.ndim == 2:
    #     estimated_target_voice_1ch_path = "./utils/estimated_target_voice_1ch.wav"
    #     sf.write(estimated_target_voice_1ch_path, estimated_target_voice_data[:, 0], args.sample_rate)
    #     estimated_voice_recog_text = asr_julius(estimated_target_voice_1ch_path)
    #     os.remove(estimated_target_voice_1ch_path)
    # else:    
    #     estimated_voice_recog_text = asr_julius(mixed_audio_path)
    # estimated_voice_recog_text = estimated_voice_recog_text.split()
    # # WERを計算
    # clean_recog_result_save_path = os.path.join(recog_result_dir, 'clean.txt')
    # mix_recog_result_save_path = os.path.join(recog_result_dir, 'mix.txt')
    # est_recog_result_save_path = os.path.join(recog_result_dir, 'est.txt')
    # wer_clean = asr_eval(reference_label_text, target_voice_recog_text, clean_recog_result_save_path)
    # wer_mix = asr_eval(reference_label_text, mixed_audio_recog_text, mix_recog_result_save_path)
    # wer_est = asr_eval(reference_label_text, estimated_voice_recog_text, est_recog_result_save_path)

    print("============================音源分離性能===============================")
    print("SDR_mix: {:.3f}, SIR_mix: {:.3f}, SAR_mix: {:.3f}".format(sdr_mix, sir_mix, sar_mix))
    print("SDR_est: {:.3f}, SIR_est: {:.3f}, SAR_est: {:.3f}".format(sdr_est, sir_est, sar_est))
    print("============================音声認識性能===============================")
    print("WER_clean: {:.3f}".format(wer_clean))
    print("WER_mix: {:.3f}".format(wer_mix))
    print("WER_est: {:.3f}".format(wer_est))


if __name__ == "__main__":
    main()
