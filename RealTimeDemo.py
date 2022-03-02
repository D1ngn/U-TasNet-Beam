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
import sys
import numpy as np
import argparse
import time as tm
import queue
import sounddevice as sd
import soundfile as sf
import pyroomacoustics as pa # 音源定位用
import threading # 録音と音声処理を並列で実行するため
import requests
from io import BytesIO
from multiprocessing import Pool

# マスクビームフォーマ関連
from models import MCComplexUnet, MCConvTasNet # 雑音・残響除去モデル、話者分離モデル
from beamformer import estimate_covariance_matrix_sig, condition_covariance, estimate_steering_vector, mvdr_beamformer, gev_beamformer, ds_beamformer, mwf, mvdr_beamformer_two_speakers
from utils.utilities import AudioProcessForComplex
from loss_func import solve_inter_channel_permutation_problem # マルチチャンネル話者分離時に使用
# 話者識別用モデル
from utils.embedder import SpeechEmbedder

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

# 一定の長さに分割された音声を繰り返し処理する関数
def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # 音声処理を別スレッドで実行するためのインスタンスを生成（並列処理しないと音が途切れる）
    # Fancy indexing with mapping creates a (necessary!) copy
    th = threading.Thread(target=worker, args=(indata.copy(),))
    # スレッドの開始
    th.start()

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
def worker(mixed_audio_data):
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
            
        # モデルに入力できるように音声をミニバッチに分けながら振幅＋位相スペクトログラムに変換
        # torch.stftを使用する場合
        mixed_audio_data_for_model_input = torch.transpose(torch.from_numpy(mixed_audio_data).float(), 0, 1)
        mixed_audio_data_for_model_input = mixed_audio_data_for_model_input.to(device) # モデルをCPUまたはGPUへ
        """mixed_audio_data_for_model_input: (num_channels, num_samples)"""
        # start_time_denoiser = time.perf_counter()
        mixed_amp_phase_spec_batch = audio_processor.preprocess_mask_estimator(mixed_audio_data_for_model_input, args.chunk_size)
        """amp_phase_spec_batch: (batch_size, num_channels, freq_bins, time_frames, real_imaginary)"""
        # 発話とそれ以外の雑音の時間周波数マスクを推定
        speech_amp_phase_spec_output, noise_amp_phase_spec_output = denoising_model(mixed_amp_phase_spec_batch)
        """speech_amp_phase_spec_output: (batch_size, num_channels, freq_bins, time_frames, real_imaginary), 
        noise_amp_phase_spec_output: (batch_size, num_channels, freq_bins, time_frames, real_imaginary)"""
        # ミニバッチに分けられた振幅＋位相スペクトログラムを時間方向に結合
        multichannel_speech_amp_phase_spec= audio_processor.postprocess_mask_estimator(mixed_complex_spec, speech_amp_phase_spec_output, args.chunk_size, args.target_aware_channel)
        """multichannel_speech_amp_phase_spec: (num_channels, freq_bins, time_frames, real_imaginary)"""
        multichannel_noise_amp_phase_spec = audio_processor.postprocess_mask_estimator(mixed_complex_spec, noise_amp_phase_spec_output, args.chunk_size, args.noise_aware_channel)
        """multichannel_noise_amp_phase_spec: (num_channels, freq_bins, time_frames, real_imaginary)"""
        # 発話のマルチチャンネルスペクトログラムを音声波形に変換
        multichannel_denoised_data = torch.istft(multichannel_speech_amp_phase_spec, n_fft=512, hop_length=160, \
                                                    normalized=True, length=mixed_audio_data.shape[0], return_complex=False)
        """multichannel_denoised_data: (num_channels, num_samples)"""
        # 雑音のマルチチャンネルスペクトログラムを音声波形に変換
        multichannel_noise_data = torch.istft(multichannel_noise_amp_phase_spec, n_fft=512, hop_length=160, \
                                                    normalized=True, length=mixed_audio_data.shape[0], return_complex=False)
        """multichannel_noise_data: (num_channels, num_samples)"""
        # 話者分離モデルに入力できるようにバッチサイズの次元を追加
        multichannel_denoised_data = torch.unsqueeze(multichannel_denoised_data, 0)
        """multichannel_denoised_data: (batch_size, num_channels, num_samples)"""
        # 話者分離
        separated_audio_data = speaker_separation_model(multichannel_denoised_data)
        """separated_audio_data: (batch_size, num_speakers, num_channels, num_samples)"""
        # チャンネルごとに順序がばらばらな発話の順序を揃える
        separated_audio_data = solve_inter_channel_permutation_problem(separated_audio_data)
        """separated_audio_data: (batch_size, num_speakers, num_channels, num_samples)"""
        # PyTorchのテンソルをNumpy配列に変換
        separated_audio_data = separated_audio_data.cpu().detach().numpy().copy() # CPU
        # バッチの次元を消して転置
        separated_audio_data = np.transpose(np.squeeze(separated_audio_data, 0), (0, 2, 1))
        """separated_audio_data: (num_speakers, num_samples, num_channels)"""
        # 分離音から目的話者の発話を選出（何番目の発話が目的話者のものかを判断） →いずれはspeaker_selectorに統一する TODO
        target_speaker_id, speech_complex_spec_all = audio_processor.speaker_selector_sig_ver(separated_audio_data, ref_dvec, embedder, device)
        """speech_complex_spec_all: (num_speakers, num_channels, freq_bins, time_frames)"""
        # 目的話者の発話の複素スペクトログラムを取得
        multichannel_target_complex_spec = speech_complex_spec_all[target_speaker_id]
        """multichannel_target_complex_spec: (num_channels, freq_bins, time_frames)"""
        multichannel_interference_complex_spec = np.zeros_like(multichannel_target_complex_spec)
        # 干渉話者の発話の複素スペクトログラムを取得
        for id in range(speech_complex_spec_all.shape[0]):
            # 目的話者以外の話者の複素スペクトログラムを足し合わせる
            if id == target_speaker_id:
                pass
            else:
                multichannel_interference_complex_spec += speech_complex_spec_all[id]
        """multichannel_interference_complex_spec: (num_channels, freq_bins, time_frames)"""
        # PyTorchのテンソルをnumpy配列に変換
        multichannel_noise_data = multichannel_noise_data.cpu().detach().numpy().copy() # CPU
        """multichannel_noise_data: (num_channels, num_samples)"""
        # 雑音の複素スペクトログラムを算出
        multichannel_noise_complex_spec = audio_processor.calc_complex_spec(multichannel_noise_data.T)
        """multichannel_noise_complex_spec: (num_channels, freq_bins, time_frames)""" 

        # 目的音のマスクと雑音のマスクからそれぞれの空間共分散行列を推定
        target_covariance_matrix = estimate_covariance_matrix_sig(multichannel_target_complex_spec)
        interference_covariance_matrix = estimate_covariance_matrix_sig(multichannel_interference_complex_spec)
        noise_covariance_matrix = estimate_covariance_matrix_sig(multichannel_noise_complex_spec)
        noise_covariance_matrix = condition_covariance(noise_covariance_matrix, 1e-6) # これがないと性能が大きく落ちる（雑音の共分散行列のみで良い）
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
        # finish_time = tm.perf_counter()
        # print("処理時間：", finish_time - start_time)
        # キューにデータを格納
        q.put(multichannel_estimated_target_voice_data)
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
        q.put(mixed_audio_data)


if __name__ == "__main__":

    # コマンドライン引数を受け取る
    parser = argparse.ArgumentParser(description='Real time voice separation')
    parser.add_argument('-dm', '--denoising_mode', action='store_true', help='whether model denoises audio or not')
    parser.add_argument('-d', '--device', type=int_or_str, default=0, help='input device (numeric ID or substring)')
    parser.add_argument('-mg', '--mic_gain', type=int, default=1, help='Microphone gain')
    parser.add_argument('-sr', '--sample_rate', type=int, default=16000, help='sampling rate')
    parser.add_argument('-cs', '--chunk_size', type=int, default=48000, help='chunk size of mask estimator and beamformer input')
    parser.add_argument('-c', '--channels', type=int, default=8, help='number of input channels')
    parser.add_argument('-fs', '--fft_size', type=int, default=512, help='size of fast fourier transform')
    parser.add_argument('-hl', '--hop_length', type=int, default=160, help='number of audio samples between adjacent STFT columns')
    parser.add_argument('-dmt', '--denoising_model_type', type=str, default='complex_unet', help='type of denoising model (FC or BLSTM or CNN or Unet or Unet_single_mask or Unet_single_mask_two_speakers)') # 雑音（残響除去）モデルのタイプ
    parser.add_argument('-ssmt', '--speaker_separation_model_type', type=str, default='conv_tasnet', help='type of speaker separator model (conv_tasnet)') # 話者分離モデルのタイプ
    parser.add_argument('-bt', '--beamformer_type', type=str, default='MVDR', help='type of beamformer (DS or MVDR or GEV or MWF)')
    parser.add_argument('-dt', '--dereverb_type', type=str, default='None', help='type of dereverb algorithm (None or WPE)')
    parser.add_argument('-ep', '--embedder_path', type=str, default="./utils/embedder.pt", help='path of pretrained embedder model')
    parser.add_argument('-rsp', '--ref_speech_path', type=str, default="./utils/ref_speech/sample.wav", help='path of reference speech')
    parser.add_argument('-tac', '--target_aware_channel', type=int, default=0, help='microphone channel near target source')
    parser.add_argument('-nac', '--noise_aware_channel', type=int, default=4, help='microphone channel near noise source')
    args = parser.parse_args()

    ########################################################音源定位用設定############################################################
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
    ###################################################################################################################################

    # GPUが使える場合はGPUを使用、使えない場合はCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：" , device)

    # JuliusサーバのURL
    url = "http://192.168.10.101:8000/asr_julius"
    # url = "http://192.168.10.116:8000/asr_julius" # Jetson Xavierではこっちを使用
    ses = requests.Session()

    wave_dir = "./audio_data/rec/"
    os.makedirs(wave_dir, exist_ok=True)
    estimated_voice_path = os.path.join(wave_dir, "record.wav")
    # ネットワークモデルの定義、チャンネルの選び方の指定、モデル入力時にパディングを行うか否かを指定
    # 雑音（残響）除去モデル
    if args.denoising_model_type == 'complex_unet':
        checkpoint_path_for_denoising_model = "./ckpt/ckpt_NoisySpeechDataset_multi_wav_test_original_length_ComplexUnet_ch_constant_snr_loss_multisteplr00001start_20210922/ckpt_epoch490.pt" # Complex U-Net speech and noise output ch constant snr loss (signal base newest model)
        denoising_model = MCComplexUnet()
        channel_select_type = 'single'
        padding = True
    else:
        print("Please specify the correct denoising model type")
    # 話者分離モデル
    if args.speaker_separation_model_type == 'conv_tasnet':
        checkpoint_path_for_speaker_separation_model = "./ckpt/ckpt_NoisySpeechDataset_multi_wav_for_ConvTasnet_snr_loss_multisteplr00001start_for_TensorRT_20211117/ckpt_epoch480.pt"
        speaker_separation_model = MCConvTasNet()
    else:
        print("Please specify the correct speaker separator type")

    # 音声処理クラスのインスタンスを作成
    # audio_processor = AudioProcess(args.sample_rate, args.fft_size, args.hop_length, channel_select_type, padding)
    audio_processor = AudioProcessForComplex(args.sample_rate, args.fft_size, args.hop_length, channel_select_type, padding)
    
    # 学習済みのパラメータをロード
    denoising_model_params = torch.load(checkpoint_path_for_denoising_model, map_location=device)
    denoising_model.load_state_dict(denoising_model_params['model_state_dict'])
    denoising_model.to(device) # モデルをCPUまたはGPUへ
    denoising_model.eval() # ネットワークを推論モードへ
    # print("モデルのパラメータ数：", count_parameters(model))
    # # 入力サンプルとともにTensorRTに変換
    # tmp = torch.ones((1, args.channels, int(args.fft_size/2)+1, 513, 2)).to(device)
    # print(tmp.shape)
    # # denoising_model = torch2trt(denoising_model, [tmp])
    # denoising_model = torch2trt(denoising_model, [tmp], fp16_mode=True) # 精度によってモード切り替え
    # 話者分離モデルの学習済みパラメータをロード
    speaker_separation_model_params = torch.load(checkpoint_path_for_speaker_separation_model, map_location=device)
    speaker_separation_model.load_state_dict(speaker_separation_model_params['model_state_dict'])
    speaker_separation_model.to(device) # モデルをCPUまたはGPUへ
    speaker_separation_model.eval() # ネットワークを推論モードへ
    # # 入力サンプルとともにTensorRTに変換
    # tmp_2 = torch.ones((1, args.channels, 48000)).to(device)
    # # denoising_model = torch2trt(denoising_model, [tmp])
    # speaker_separation_model = torch2trt(speaker_separation_model, [tmp_2], fp16_mode=True) # 精度によってモード切り替え
    # 話者識別モデルの学習済みパタメータをロード（いずれはhparamsでパラメータを指定できる様にする TODO）
    embedder = SpeechEmbedder()
    embed_params = torch.load(args.embedder_path, map_location=device)
    embedder.load_state_dict(embed_params)
    embedder.to(device) # モデルをCPUまたはGPUへ
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
    # PyTorchのテンソルからnumpy配列に変換
    ref_dvec = ref_dvec.cpu().detach().numpy().copy() # CPU

    # 録音＋音声認識
    try:        
        # 参考： 「https://python-sounddevice.readthedocs.io/en/0.3.12/examples.html#recording-with-arbitrary-duration」
        q = queue.Queue() # データを格納した順に取り出すことができるキューを作成
        # Make sure the file is opened before recording anything
        with sf.SoundFile(estimated_voice_path, mode='w', samplerate=args.sample_rate,
                        channels=args.channels) as file:
            with sd.InputStream(samplerate=args.sample_rate, blocksize=args.chunk_size, device=args.device,
                                channels=args.channels, callback=audio_callback):
                print('#' * 50)
                print('press Ctrl+C to stop the recording')
                print('#' * 50)
                while True:
                    # キューからデータを取り出してファイルに書き込み
                    audio_data = q.get() # ここでinput overflowが起きる
                    file.write(audio_data)
                    # 1ch分を取り出し、バイナリデータに変換
                    if audio_data.ndim == 2:
                        audio_data = audio_data[:, 0]
                    out = BytesIO()
                    np.save(out, audio_data)
                    binary = out.getvalue()
                    # 音声認識サーバに送信し、認識結果を受信
                    result = ses.post(url, files={'myFile': binary})
                    print("音声認識結果：", result.text)
    except KeyboardInterrupt:
        print('\nRecording finished: ' + repr(estimated_voice_path))
        parser.exit(0)
    except Exception as e:
        parser.exit(type(e).__name__ + ': ' + str(e))