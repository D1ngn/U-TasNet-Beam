# 必要モジュールのimport
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchaudio

import os
import numpy as np
import soundfile as sf
import time
import argparse
import pandas as pd
import glob 
import datetime

from models import MCComplexUnet
from utils.loss_func import snr_loss, si_snr_loss

# PyTorch 以外のRNGを初期化
# random.seed(0)
np.random.seed(0)
# PyTorch のRNGを初期化
torch.manual_seed(0)

# データセットのクラス
class SpeechDataset(data.Dataset):
    def __init__(self, mixed_wav_list, target_wav_list, noise_wav_list, \
                 sample_rate, fft_size, hop_length):
        self.mixed_wav_list = mixed_wav_list # 混合音声のファイルリスト
        self.target_wav_list = target_wav_list # 目的音のファイルリスト
        self.noise_wav_list = noise_wav_list # 雑音のファイルリスト
        # スペクトログラム算出用
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        # スペクトログラムの時間フレーム数
        self.max_time_frames = 513

    # マルチチャンネル音声のロード
    def _load_audio(self, file_path):
        waveform, _ = torchaudio.load(file_path)
        """waveform: (num_channels, num_samples)"""
        return waveform
    
    # マルチチャンネルの振幅スペクトログラムと位相スペクトログラムを算出
    def _calc_amp_phase_spec(self, waveform):
        """waveform: (num_channels, num_samples)"""
        multi_amp_phase_spec = torch.stft(input=waveform, n_fft=self.fft_size, hop_length=self.hop_length, normalized=True, return_complex=False)
        """multi_amp_phase_spec: (num_channels, freq_bins, time_frames, real_imaginary)"""
        return multi_amp_phase_spec
    
    # マルチチャンネルの複素スペクトログラムを時間フーレム方向に0パディング
    def _zero_pad_spec(self, complex_spec):
        """complex_spec: (num_channels, freq_bins, time_frames, real-imaginary)"""
        complex_spec_padded = nn.ZeroPad2d((0, 0, 0, self.max_time_frames-complex_spec.shape[2]))(complex_spec)
        """complex_spec: (num_channels, freq_bins, time_frames=self.max_time_frames, real_imaginary)"""
        return complex_spec_padded
    
    def __len__(self):
        return len(self.mixed_wav_list)

    def __getitem__(self, index):
        # ファイルパスを取得
        mixed_wav_path = self.mixed_wav_list[index]
        target_wav_path = self.target_wav_list[index]
        noise_wav_path = self.noise_wav_list[index]
        # 音声データをロード
        mixed_audio_data = self._load_audio(mixed_wav_path)
        """mixed_audio_data: (num_channels, num_samples)"""
        target_audio_data = self._load_audio(target_wav_path)
        """target_audio_data: (num_channels, num_samples)"""
        noise_audio_data = self._load_audio(noise_wav_path)
        """noise_audio_data: (num_channels, num_samples)"""
        # 音声波形データを振幅スペクトログラム＋位相スペクトログラムに変換
        mixed_amp_phase_spec = self._calc_amp_phase_spec(mixed_audio_data)
        """mixed_amp_phase_spec: (num_channels, freq_bins, time_steps, real_imaginary)"""
        target_amp_phase_spec = self._calc_amp_phase_spec(target_audio_data)
        """target_amp_phase_spec: (num_channels, freq_bins, time_steps, real_imaginary)"""
        noise_amp_phase_spec = self._calc_amp_phase_spec(noise_audio_data)
        """noise_amp_phase_spec: (num_channels, freq_bins, time_steps, real_imaginary)"""
        # モデルのサイズに合わせてパディング
        mixed_amp_phase_spec = self._zero_pad_spec(mixed_amp_phase_spec)
        """mixed_amp_phase_spec: (num_channels, freq_bins, time_steps, real_imaginary)"""
        target_amp_phase_spec = self._zero_pad_spec(target_amp_phase_spec)
        """target_complex_spec: (num_channels, freq_bins, time_steps, real_imaginary)"""
        noise_amp_phase_spec = self._zero_pad_spec(noise_amp_phase_spec)
        """noise_complex_spec: (num_channels, freq_bins, time_steps, real_imaginary)"""
        return mixed_amp_phase_spec, target_amp_phase_spec, noise_amp_phase_spec

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
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    log_epoch = checkpoint['log_epoch']
    print("{}からデータをロードしました。エポック{}から学習を再開します。".format(checkpoint_path, start_epoch))
    return start_epoch, model, optimizer, log_epoch

# 振幅＋位相スペクトログラムを音声波形に変換
def spec_to_waveform(spec):
    """
    spec: (batch_size, num_channels, freq_bins, time_steps, real_imaginary)
    """
    # バッチサイズの次元とチャンネル数の次元を掛け合わせる（一括処理）
    spec = spec.contiguous().view(-1, spec.shape[2], spec.shape[3], spec.shape[4])
    """spec: (batch_size*num_channels, freq_bins, time_steps, real_imaginary)"""
    # 逆短時間フーリエ変換により音声波形データへ変換
    waveform = torch.istft(spec, n_fft=512, hop_length=160, normalized=True)
    """waveform: (batch_size*num_channels, num_samples)"""
    return waveform

# モデルを学習させる関数を作成（引数はargsかconfigでまとめて渡すようにする） TODO
def train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs, param_save_dir, checkpoint_path=None):
# def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, param_save_dir, channel_select_type, checkpoint_path=None):
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

    # 学習を再開する場合はパラメータをロード、最初から始める場合は特に処理は行われない
    if checkpoint_path is not None:
        start_epoch, model, optimizer, log_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)
        # GPU環境で学習したOptimizerを再度GPU環境で学習させる場合は逐一値をdeviceへ送る
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
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
            for mixed_amp_phase_spec, target_amp_phase_spec, noise_amp_phase_spec in dataloaders_dict[phase]:
                """
                mixed_amp_phase_spec: (batch_size, num_channels, freq_bins, time_steps, real_imaginary)
                target_amp_phase_spec: (batch_size, num_channels, freq_bins, time_steps, real_imaginary)
                noise_amp_phase_spec: (batch_size, num_channels, freq_bins, time_steps, real_imaginary)
                """
                # GPUが使える場合、データをGPUへ送る
                mixed_amp_phase_spec = mixed_amp_phase_spec.to(device)
                target_amp_phase_spec = target_amp_phase_spec.to(device)
                noise_amp_phase_spec = noise_amp_phase_spec.to(device)
                # optimizerを初期化
                optimizer.zero_grad()
                # 順伝播
                with torch.set_grad_enabled(phase == 'train'):
                    # 混合音のスペクトログラムをモデルに入力し、目的音のマスクと雑音のマスクを推定
                    estimated_speech_amp_phase_spec, estimated_noise_amp_phase_spec = model(mixed_amp_phase_spec)
                    
                    # 振幅＋位相スペクトログラムを音声波形に変換
                    speech = spec_to_waveform(target_amp_phase_spec)
                    """speech: (batch_size*num_channels, num_samples)"""
                    noise = spec_to_waveform(noise_amp_phase_spec)
                    """noise: (batch_size*num_channels, num_samples)"""
                    est_speech = spec_to_waveform(estimated_speech_amp_phase_spec)
                    """est_speech: (batch_size*num_channels, num_samples)"""
                    est_noise = spec_to_waveform(estimated_noise_amp_phase_spec)
                    """est_noise: (batch_size*num_channels, num_samples)"""
                    
                    # 損失を計算
                    # SNR(SDR) or SI-SNR(SI-SDR)
                    loss = criterion(speech, est_speech) + criterion(noise, est_noise)

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
        
        # 学習率の管理
        scheduler.step(epoch_val_loss)

        # 学習経過を分析できるようにcsvファイルにログを保存 → tensorboardに変更しても良いかも
        log_epoch = {'epoch': epoch+1, 'train_loss': epoch_train_loss/num_train_data, 'val_loss': epoch_val_loss/num_val_data}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        log_save_path = os.path.join(param_save_dir, "log.xlsx")
        df.to_excel(log_save_path, index=False)

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
    sample_rate = 16000 # サンプリングレート
    num_channels = 8 # 音声のチャンネル数
    batch_size = 64 # バッチサイズ
    fft_size = 512 # 短時間フーリエ変換のフレーム長
    hop_length = 160 # 短時間フーリエ変換においてフレームをスライドさせる幅
    model_type = 'Complex_Unet'
    # ネットワークモデルの定義、チャンネルの選び方の指定、モデル入力時にパディングを行うか否かを指定
    if model_type == 'Complex_Unet':
        model = MCComplexUnet()
        padding = False
    
    # データセットを作成
    dataset_dir = "../data/NoisySpeechDataset_multi_wav_test_original_length_20210526/" # Jupyterhub用
    # dataset_dir = "../data/NoisySpeechDataset_multi_wav_test_original_length_rt0300_20210702/"
    # dataset_dir = "../AudioDatasets/NoisySpeechDatabase/"
    train_mixed_wav_list, train_target_wav_list, train_noise_wav_list, \
        val_mixed_wav_list, val_target_wav_list, val_noise_wav_list = mk_datapath_list(dataset_dir)
    # データセットのインスタンスを作成
    train_dataset = SpeechDataset(train_mixed_wav_list, train_target_wav_list, train_noise_wav_list, \
        sample_rate, fft_size, hop_length)
    val_dataset = SpeechDataset(val_mixed_wav_list, val_target_wav_list, val_noise_wav_list, \
        sample_rate, fft_size, hop_length)
    # データローダーを作成
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloaders_dict = {'train':train_dataloader, 'val':val_dataloader} # データローダーを格納するリスト
    # 損失関数を定義
    # criterion = nn.MSELoss(reduction='mean') # MSELoss(input, target) : inputとtargetの各要素の差の2乗の平均
    criterion = snr_loss
    # 最適化手法を定義
    # optimizer = optim.Adam(model.parameters(), lr=0.001) # Default
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # val lossが振動してしまうので学習率を下げる
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) # val loss が3エポックの間下がらなかった場合学習率を0.5倍にする
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5) # 10エポックごとに学習率を0.5倍にする
    # 各種設定 → いずれ１つのファイルからデータを読み込ませたい
    num_epochs = 500 # epoch数を指定
    # 学習済みモデルのパラメータを保存するディレクトリを作成
    param_save_dir = "./ckpt/ckpt_NoisySpeechDataset_multi_wav_test_original_length_ComplexUnet_ch_constant_snr_loss_multisteplr00001start_20210922" # 学習済みモデルのパラメータを保存するディレクトリのパスを指定
    os.makedirs(param_save_dir, exist_ok=True)
    # モデルを学習
    train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs, param_save_dir, checkpoint_path=args.checkpoint_path)
