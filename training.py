# 必要モジュールのimport
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import librosa
import time
import argparse
import pandas as pd
import glob
import pickle as pl
import datetime
import wave

from models import FCMaskEstimator, BLSTMMaskEstimator, UnetMaskEstimator_kernel3
from utils import load_audio_file, wave_to_spec

# PyTorch 以外のRNGを初期化
# random.seed(0)
np.random.seed(0)
# PyTorch のRNGを初期化
torch.manual_seed(0)
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# データの前処理を行うクラス
class AudioProcess():
    def __init__(self, sample_rate, fft_size, hop_length, model_type):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.model_type = model_type
        # ログメルスペクトログラムを使用する場合
        self.num_mels = 40
        # U-Netを使用する場合
        self.spec_time_frames = 513
    
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
        """multi_complex_spec: (num_channels, freq_bins, time_steps)"""
        return multi_complex_spec

    # 振幅スペクトログラムを算出
    def calc_amp_spec(self, complex_spec):
        """complex_spec: (num_channels, freq_bins, time_steps)"""
        amp_spec = np.zeros_like(complex_spec, dtype=np.float32)
        for i in range(complex_spec.shape[0]):
            amp_spec[i] = np.abs(complex_spec[i])
        """amp_spec: (num_channels, freq_bin, time_steps)"""
        return amp_spec
    
    # 位相スペクトログラムを算出
    def calc_phase_spec(self, complex_spec):
        """complex_spec: (num_channels, freq_bins, time_steps)"""
        phase_spec = np.zeros_like(complex_spec, dtype=np.float32)
        for i in range(complex_spec.shape[0]):
            phase_spec[i] = np.angle(complex_spec[i]) # ラジアン単位（-π ~ +π）
        """amp_spec: (num_channels, freq_bin, time_steps)"""
        return phase_spec
    
    # ログスペクトログラムを算出 TODO
    def calc_log_spec(self, complex_spec):
        """
        complex_spec: (num_channels, freq_bins, time_steps)
        """
        log_spec = np.zeros_like(complex_spec, dtype='float32')
        # チャンネルごとにログスペクトログラムに変換
        for i in range(complex_spec.shape[0]):
            power_spec = np.abs(complex_spec[i]) ** 2 # パワースペクトログラムを算出
            log_spec[i] = 10.0 * np.log10(np.maximum(1e-10, power_spec)) # logがマイナス無限にならないように対数変換
        """log_spec: (num_channels, num_mels, time_steps)"""
        return log_spec
    
    # ログメルスペクトログラムを算出（使用するかは要検討）TODO
    def calc_log_mel_spec(self, complex_spec):
        """
        complex_spec: (num_channels, freq_bins, time_steps)
        """
        # メルフィルタバンクを生成
        mel_fb = librosa.filters.mel(self.sample_rate, self.fft_size, n_mels=self.num_mels)
        log_mel_spec = np.zeros((complex_spec.shape[0], self.num_mels, complex_spec.shape[2]), dtype='float32')
        # チャンネルごとにログメルスペクトログラムに変換
        for i in range(complex_spec.shape[0]):
            power_spec = np.abs(complex_spec[i]) ** 2 # パワースペクトログラムを算出
            mel_power_spec = np.dot(mel_fb, power_spec) # メルフィルタバンクをかける
            log_mel_spec[i] = 10.0 * np.log10(np.maximum(1e-10, mel_power_spec)) # logがマイナス無限にならないように対数変換
        """log_mel_spec: (num_channels, num_mels, time_steps)"""
        return log_mel_spec
    
    def __call__(self, wav_data):
        # # 音声波形データをロード
        # wav_data = load_audio_file(data_path, self.audio_length, self.sample_rate)
        # """wav_data: (num_samples, num_channels)"""

        # マルチチャンネル音声データをスペクトログラムに変換
        multi_complex_spec = self.calc_complex_spec(wav_data)
        """multi_complex_spec: (num_channels, freq_bins=, time_steps)"""
        # U-Netの場合 
        if self.model_type == 'Unet':
            # モデルの入力サイズに合わせて、スペクトログラムの後ろの部分を0埋め(パディング)
            # データが三次元の時、(手前, 奥),(上,下), (左, 右)の順番でパディングを追加
            multi_complex_spec = np.pad(multi_complex_spec, [(0, 0), (0, 0), (0, self.spec_time_frames-multi_complex_spec.shape[2])], 'constant')
            """multi_complex_spec: (num_channels=8, freq_bins=257, time_steps=513)"""
        # 振幅スペクトログラムを算出
        multi_amp_spec = self.calc_amp_spec(multi_complex_spec)
        """multi_amp_spec: (num_channels, freq_bins, time_steps)"""
        # # ログスペクトログラムを算出
        # multi_log_spec = self.calc_log_spec(multi_complex_spec)
        # """multi_log_spec: (num_channels, freq_bins, time_steps)"""
        # # 位相スペクトログラムを算出
        # multi_phase_spec = self.calc_phase_spec(multi_complex_spec)
        # """multi_phase_spec: (num_channels, freq_bins, time_steps)"""
        # # 振幅スペクトログラムの代わりにログメルスペクトログラムを使用する場合 TODO
        # multi_log_mel_spec = self.calc_log_mel_spec(multi_complex_spec)
        # return multi_complex_spec, multi_amp_spec, multi_log_mel_spec

        return multi_complex_spec, multi_amp_spec
        # return multi_complex_spec, multi_log_spec

# データセットのクラス
class IRMDataset(data.Dataset):
    def __init__(self, mixed_spec_list, target_spec_list, interference_spec_list, transform, model_type, \
        audio_length, sample_rate, num_channels, fft_size, num_mels=20, target_aware_channel=0, noise_aware_channel=4):
        self.mixed_spec_list = mixed_spec_list # 混合音声のファイルリスト
        self.target_spec_list = target_spec_list # 目的音のファイルリスト
        self.interference_spec_list = interference_spec_list # 干渉音のファイルリスト
        self.transform = transform
        self.model_type = model_type
        self.target_aware_channel = target_aware_channel # 目的音に近い位置にあるチャンネル TODO
        self.noise_aware_channel = noise_aware_channel # 雑音に近い位置にあるチャンネル TODO
        self.audio_length = audio_length
        # ログメルスペクトログラム算出用（使用するかは要検討）
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.fft_size = fft_size
        self.num_mels = num_mels
    
    # データを標準化（平均0、分散1に正規化（Z-score Normalization））
    def standardize(self, data):
        data_mean = data.mean(keepdims=True)
        data_std = data.std(keepdims=True, ddof=0) # 母集団の標準偏差（標本標準偏差を使用する場合はddof=1）
        standardized_data = (data - data_mean) / data_std
        return standardized_data

    # Ideal Ratio Maskを算出（cIRMに変更するのもあり） TODO
    def calc_ideal_ratio_mask(self, target_spec, noise_spec):
        """
        target_spec: (num_channels, freq_bins, time_steps)
        noise_spec: (num_channels, freq_bins, time_steps)
        """
        # 参考：「https://gist.github.com/jonashaag/677e1ddab99f3daba367de9ec022e942#file-cirm-py-L39」
        # 0除算を避ける
        target_IRM = np.sqrt(target_spec ** 2 / np.maximum((target_spec ** 2 + noise_spec ** 2), 1e-7))
        noise_IRM = np.sqrt(noise_spec ** 2 / np.maximum((target_spec ** 2 + noise_spec ** 2), 1e-7))
        return target_IRM, noise_IRM

    def __len__(self):
        return len(self.mixed_spec_list)

    def __getitem__(self, index):
        # ファイルパスを取得
        mixed_wav_path = self.mixed_spec_list[index]
        target_wav_path = self.target_spec_list[index]
        interference_wav_path = self.interference_spec_list[index]

        # 音声波形データをロード
        mixed_wav_data = load_audio_file(mixed_wav_path, self.audio_length, self.sample_rate)
        """mixed_wav_data: (num_samples, num_channels)"""
        target_wav_data = load_audio_file(target_wav_path, self.audio_length, self.sample_rate)
        """target_wav_data: (num_samples, num_channels)"""
        interference_wav_data = load_audio_file(interference_wav_path, self.audio_length, self.sample_rate)
        """interference_wav_data: (num_samples, num_channels)"""

        # 音声波形データを振幅スペクトログラムに変換
        _, mixed_amp_spec = self.transform(mixed_wav_data)
        """mixed_amp_spec: (num_channels, freq_bins, time_steps)"""
        _, target_amp_spec = self.transform(target_wav_data)
        """target_amp_spec: (num_channels, freq_bins, time_steps)"""
        _, interference_amp_spec = self.transform(interference_wav_data)
        """interference_amp_spec: (num_channels, freq_bins, time_steps)"""

        # # 音声波形データをログスペクトログラムに変換
        # _, mixed_log_spec = self.transform(mixed_wav_path)
        # """mixed_log_spec: (num_channels, freq_bins, time_steps)"""
        # _, target_log_spec = self.transform(target_wav_path)
        # """target_log_spec: (num_channels, freq_bins, time_steps)"""
        # _, interference_log_spec = self.transform(interference_spec_path)
        # """interference_log_spec: (num_channels, freq_bins, time_steps)"""

        # IRM(Ideal Ratio Mask)を算出（教師データ）
        target_IRM, noise_IRM = self.calc_ideal_ratio_mask(target_amp_spec, interference_amp_spec)
        # target_IRM, noise_IRM = self.calc_ideal_ratio_mask(target_log_spec, interference_log_spec)
        """target_IRM: (num_channels, freq_bins, time_steps), noise_IRM: (num_channels, freq_bins, time_steps)"""
        # 複数チャンネルのうち1チャンネル分のマスクを算出
        if self.model_type == 'FC' or 'Unet':
            # 目的音と干渉音に近いチャンネルのマスクをそれぞれ使用（選択するチャンネルを変えて実験してみる） TODO
            target_IRM = target_IRM[self.target_aware_channel, :, :]
            noise_IRM = noise_IRM[self.noise_aware_channel, :, :]
        elif self.model_type == 'BLSTM':
            # 複数チャンネル間のマスク値の中央値をとる（median pooling）
            (target_IRM, _) = torch.median(target_IRM, dim=0)
            (noise_IRM, _) = torch.median(noise_IRM, dim=0)
        """target_IRM: (freq_bins, time_steps), noise_IRM: (freq_bins, time_steps)"""

        # スペクトログラムを標準化
        mixed_amp_spec = self.standardize(mixed_amp_spec)
        # numpy形式のデータをpytorchのテンソルに変換
        mixed_amp_spec = torch.from_numpy(mixed_amp_spec.astype(np.float32)).clone()
        target_IRM = torch.from_numpy(target_IRM.astype(np.float32)).clone()
        noise_IRM = torch.from_numpy(noise_IRM.astype(np.float32)).clone()

        # # ログメルスペクトログラムの場合
        # # numpy形式のデータをpytorchのテンソルに変換
        # mixed_log_spec = torch.from_numpy(mixed_log_spec.astype(np.float32)).clone()
        # target_IRM = torch.from_numpy(target_IRM.astype(np.float32)).clone()
        # noise_IRM = torch.from_numpy(noise_IRM.astype(np.float32)).clone()

        return mixed_amp_spec, target_IRM, noise_IRM
        # return mixed_log_spec, target_IRM, noise_IRM

# trainデータとvalデータのファイルパスリストを取得
def mk_datapath_list(dataset_dir):
    # trainデータのパスとラベルを取得
    train_mixed_path_template = os.path.join(dataset_dir, "train/*_mixed.wav")
    train_mixed_wav_list = glob.glob(train_mixed_path_template) # 混合音声のパスリスト
    train_target_wav_list = [] # 目的音のパスリスト
    train_interference_wav_list = [] # 干渉音のパスリスト
    for train_mixed_wav_path in train_mixed_wav_list:
        train_mixed_file_name = os.path.basename(train_mixed_wav_path) # (例)p226_001_mixed.wav
        # 目的音
        train_target_file_name = train_mixed_file_name.rsplit('_', maxsplit=1)[0] + "_target.wav" # (例)p226_001_target.wav
        train_target_file_path = os.path.join(dataset_dir, "train/{}".format(train_target_file_name))
        train_target_wav_list.append(train_target_file_path)
        # 干渉音
        train_interference_file_name = train_mixed_file_name.rsplit('_', maxsplit=1)[0] + "_interference.wav" # (例)p226_001_interference.wav
        train_interference_file_path = os.path.join(dataset_dir, "train/{}".format(train_interference_file_name))
        train_interference_wav_list.append(train_interference_file_path)

    # valデータのパスとラベルを取得
    val_mixed_path_template = os.path.join(dataset_dir, "val/*_mixed.wav")
    val_mixed_wav_list = glob.glob(val_mixed_path_template) # 混合音声のパスリスト
    val_target_wav_list = [] # 目的音のパスリスト
    val_interference_wav_list = [] # 干渉音のパスリスト
    for val_mixed_wav_path in val_mixed_wav_list:
        val_mixed_file_name = os.path.basename(val_mixed_wav_path) # (例)p226_001_mixed.wav
        # 目的音
        val_target_file_name = val_mixed_file_name.rsplit('_', maxsplit=1)[0] + "_target.wav" # (例)p226_001_target.wav
        val_target_file_path = os.path.join(dataset_dir, "val/{}".format(val_target_file_name))
        val_target_wav_list.append(val_target_file_path)
        # 干渉音
        val_interference_file_name = val_mixed_file_name.rsplit('_', maxsplit=1)[0] + "_interference.wav" # (例)p226_001_interference.wav
        val_interference_file_path = os.path.join(dataset_dir, "val/{}".format(val_interference_file_name))
        val_interference_wav_list.append(val_interference_file_path)

    return train_mixed_wav_list, train_target_wav_list, train_interference_wav_list, val_mixed_wav_list, val_target_wav_list, val_interference_wav_list


# 既存のチェックポイントファイルをロード
def load_checkpoint(model, optimizer, checkpoint_path, device):
    # チェックポイントファイルがない場合エラー
    assert os.path.isfile(checkpoint_path)
    # チェックポイントファイルをロード
    checkpoint = torch.load(checkpoint_path, map_location=device)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    log_epoch = checkpoint['log_epoch']
    print("{}からデータをロードしました。エポック{}から学習を再開します。".format(checkpoint_path, start_epoch))
    return start_epoch, model, optimizer, log_epoch


# モデルを学習させる関数を作成（引数はargsかconfigでまとめて渡すようにする） TODO
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, param_save_dir, model_type, checkpoint_path):

    # GPUが使える場合あはGPUを使用、使えない場合はCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：" , device)

    # モデルがある程度固定(イテレーションごとの入力サイズが一定)であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # 各カウンタを初期化
    start_epoch = 0
    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []

    # マスクのチャンネルを指定（いずれはconfigまたはargsで指定）TODO
    target_aware_channel = 0
    noise_aware_channel = 4

    # 学習を再開する場合はパラメータをロード、最初から始める場合は特に処理は行われない
    if checkpoint_path is not None:
        start_epoch, model, optimizer, log_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)
    else:
        print("checkpointファイルがありません。最初から学習を開始します。")

    # ネットワークをGPUへ
    model.to(device)

    # 学習データと検証データの数、バッチサイズを取得
    num_train_data = len(dataloaders_dict['train'].dataset)
    num_val_data = len(dataloaders_dict['val'].dataset)
    batch_size = dataloaders_dict['train'].batch_size

    print("num_train_data:", num_train_data)
    print("num_val_data:", num_val_data)
    print("batch_size:", batch_size)

    # epochごとのループ
    for epoch in range(start_epoch, num_epochs):

        # 開始時刻を記録
        epoch_start_time = time.time()
        iter_start_time = time.time()

        print("エポック {}/{}".format(epoch+1, num_epochs))

        # モデルのモードを切り替える(学習 ⇔ 検証)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # 学習モード
            else:
                model.eval() # 検証モード

            # データローダーからミニバッチずつ取り出すループ
            for mixed_amp_spec, target_IRM, noise_IRM in dataloaders_dict[phase]:
                """
                mixed_amp_spec: (batch_size, channels=8, freq_bins, time_steps)
                target_IRM: (batch_size, freq_bins, time_steps)
                noise_IRM: (batch_size, freq_bins, time_steps)
                """
                # GPUが使える場合、データをGPUへ送る
                mixed_amp_spec = mixed_amp_spec.to(device)
                target_IRM = target_IRM.to(device)
                noise_IRM = noise_IRM.to(device)
                # optimizerを初期化
                optimizer.zero_grad()
                # 順伝播
                with torch.set_grad_enabled(phase == 'train'):
                    # 混合音声のスペクトログラムをモデルに入力し、目的音のマスクと雑音のマスクを推定
                    target_mask_output, noise_mask_output = model(mixed_amp_spec)
                    """target_mask_output: (batch_size, channels=8, freq_bins, time_steps)"""
                    """noise_mask_output: (batch_size, channels=8, freq_bins, time_steps)"""
                    if model_type == "FC" or model_type == "Unet":
                        # マスクのチャンネルを指定（目的音に近いチャンネルと雑音に近いチャンネル）
                        estimated_target_mask = target_mask_output[:, target_aware_channel, :, :]
                        """estimated_target_mask: (batch_size, freq_bins, time_steps)"""
                        estimated_noise_mask = noise_mask_output[:, noise_aware_channel, :, :]
                        """estimated_noise_mask: (batch_size, freq_bins, time_steps)"""
                    elif model_type == "BLSTM":
                        # 複数チャンネル間のマスク値の中央値をとる（median pooling）
                        (estimated_target_mask, _) = torch.median(target_mask_output, dim=1)
                        """estimated_target_mask: (batch_size, freq_bins, time_steps)"""
                        (estimated_noise_mask, _) = torch.median(noise_mask_output, dim=1)
                        """estimated_noise_mask: (batch_size, freq_bins, time_steps)"""
                    else:
                        print("Please specify the correct model type")

                    # # 目的音のマスクと雑音のマスクから空間相関行列を推定
                    # target_covariance_matrix = estimate_covariance_matrix(mixed_spec, target_mask)
                    # noise_covariance_matrix = estimate_covariance_matrix(mixed_spec, noise_mask)
                    # # MVDRビームフォーマによる雑音除去を実行
                    # estimated_spec = mvdr_beamformer(mixed_spec, target_covariance_matrix, noise_covariance_matrix)
                    # # estimated_spec = max_snr_beamformer(mixed_spec, target_covariance_matrix, noise_covariance_matrix)
                      
                    # 損失を計算
                    loss = 0.5 * criterion(estimated_target_mask, target_IRM) + 0.5 * criterion(estimated_noise_mask, noise_IRM)

                    # 学習時は誤差逆伝播(バックプロパゲーション)
                    if phase == 'train':
                        # 誤差逆伝播を行い、勾配を算出
                        loss.backward()
                        # パラメータ更新
                        optimizer.step()
                        # 10iterationごとにlossと処理時間を表示
                        if (iteration % 10 == 0):
                            iter_finish_time = time.time()
                            duration_per_ten_iter = iter_finish_time - iter_start_time
                            # 0次元のテンソルから値を取り出す場合は「.item()」を使う
                            print("イテレーション {} | Loss:{:.4f} | 経過時間:{:.4f}[sec]".format(iteration, loss.item()/batch_size, duration_per_ten_iter))
                            epoch_train_loss += loss.item()

                        epoch_train_loss += loss.item()
                        iteration += 1
                    # 検証時
                    else:
                        epoch_val_loss += loss.item()

        # epochごとのlossと正解率を表示
        epoch_finish_time = time.time()
        duration_per_epoch = epoch_finish_time - epoch_start_time
        print("=" * 30)
        print("エポック {} | Epoch train Loss:{:.4f} | Epoch val Loss:{:.4f}".format(epoch+1, epoch_train_loss/num_train_data, epoch_val_loss/num_val_data))
        print("経過時間:{:.4f}[sec/epoch]".format(duration_per_epoch))

        # 学習経過を分析できるようにcsvファイルにログを保存 → tensorboardに変更しても良いかも
        log_epoch = {'epoch': epoch+1, 'train_loss': epoch_train_loss/num_train_data, 'val_loss': epoch_val_loss/num_val_data}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        log_save_path = os.path.join(param_save_dir, "log.csv")
        df.to_csv(log_save_path)

        # エポックごとのタイムログをファイルに追記
        time_log = os.path.join(param_save_dir, "time_log.txt")
        with open(time_log, mode='a') as f:
            f.write("エポック {} | {}\n".format(epoch+1, datetime.datetime.now()))

        # epochごとの損失を初期化
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # 学習したモデルのパラメータを保存
        if ((epoch+1) % 10 == 0):
            param_save_path = os.path.join(param_save_dir, "ckpt_epoch{}.pt".format(epoch+1))
            # torch.save(net.state_dict(), param_save_path) # 推論のみを行う場合
            # 学習を再開できるように変更
            torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'log_epoch': log_epoch
            }, param_save_path)


if __name__ == '__main__':

    # コマンドライン引数を受け取る
    parser = argparse.ArgumentParser(description='for unet train')
    parser.add_argument('--checkpoint_path', default=None, help="checkpoint path if you restart training")
    args = parser.parse_args()
    # 各パラメータを設定（いずれはargsまたはconfigで指定） TODO
    audio_length = 3 # 音声の長さ[sec]
    sample_rate = 16000 # サンプリングレート
    num_channels = 8 # 音声のチャンネル数
    batch_size = 2 # バッチサイズ
    fft_size = 512 # 短時間フーリエ変換のフレーム長
    hop_length = 160 # 短時間フーリエ変換においてフレームをスライドさせる幅
    # spec_freq_bins = fft_size / 2 + 1 # スペクトログラムの周波数次元数（257）
    # spec_time_frames = 513 # スペクトログラムのフレーム数 spec_freq_dim=512のとき、音声の長さが5秒の場合は128, 3秒の場合は64
    model_type = 'BLSTM' # "FC" or "BLSTM" or "Unet"
    # モデルを作成
    if model_type == 'FC':
        model = FCMaskEstimator()
    elif model_type == 'BLSTM':
        model = BLSTMMaskEstimator()
    elif model_type == 'Unet':
        model = UnetMaskEstimator_kernel3()
    # データセットを作成
    dataset_dir = "../data/NoisySpeechDataset_for_unet_fft_512_multi_wav_1130/" # jupyterhub用
    train_mixed_wav_list, train_target_wav_list, train_interference_wav_list, \
        val_mixed_wav_list, val_target_wav_list, val_interference_wav_list = mk_datapath_list(dataset_dir)
    # 前処理クラスのインスタンスを作成(numpy形式のスペクトログラムをpytorchのテンソルに変換する)
    transform = AudioProcess(sample_rate, fft_size, hop_length, model_type) 
    # データセットのインスタンスを作成
    train_dataset = IRMDataset(train_mixed_wav_list, train_target_wav_list, train_interference_wav_list, \
        transform, model_type, audio_length, sample_rate, num_channels, fft_size)
    val_dataset = IRMDataset(val_mixed_wav_list, val_target_wav_list, val_interference_wav_list, \
        transform, model_type, audio_length, sample_rate, num_channels, fft_size)
    # データローダーを作成
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloaders_dict = {'train':train_dataloader, 'val':val_dataloader} # データローダーを格納するリスト
    # テスト用
    # batch_iterator = iter(dataloaders_dict["val"])
    # mixed_spec, target_spec = next(batch_iterator)
    # print(mixed_spec.shape)
    # print(target_spec.shape)
    # 損失関数を定義
    criterion = nn.MSELoss(reduction='mean') # MSELoss(input, target) : inputとtargetの各要素の差の2乗の平均
    # 最適化手法を定義
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 各種設定 → いずれ１つのファイルからデータを読み込ませたい
    num_epochs = 300 # epoch数を指定
    # 学習済みモデルのパラメータを保存するディレクトリを作成
    param_save_dir = "./ckpt" # 学習済みモデルのパラメータを保存するディレクトリのパスを指定
    os.makedirs(param_save_dir, exist_ok=True)
    # モデルを学習
    train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, param_save_dir, model_type, checkpoint_path=args.checkpoint_path)
