import torch
import torch.nn as nn
# from torchsummary import summary

from typing import Tuple, Optional

def init_weights(layer):
    # 全結合層または畳み込み層または転置畳み込み層の場合
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(layer.weight) # xavierの重みで初期化
        # バイアスがある場合
        if layer.bias is not None:
            layer.bias.data.fill_(0.0) # 0.0で初期化

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding), # ２次元データの畳み込み演算を行う層 引数は入力のチャンネル数、出力のチャンネル数、カーネル(フィルタ)の大きさ
            nn.BatchNorm2d(out_channels), # batch normalizationを行う層 引数は入力データのチャンネル数
            nn.LeakyReLU(0.2, inplace=True) # inplace=Trueにすることで、使用メモリを削減
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class TransposeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(TransposeConvBlock, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding), # ２次元データの畳み込み演算を行う層 引数は入力のチャンネル数、出力のチャンネル数、カーネル(フィルタ)の大きさ
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=0.5), # 50%の割合でドロップアウトを実行
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.deconv(x)
        return x

class TransposeConvBlock_without_dropout(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(TransposeConvBlock_without_dropout, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.deconv(x)
        return x

class LinearNorm(nn.Module):
    def __init__(self, lstm_hidden, emb_dim):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(lstm_hidden, emb_dim)

    def forward(self, x):
        return self.linear_layer(x)

class FCBlock(nn.Module):
    def __init__(self, in_feature_dim, out_feature_dim):
        """
        in_feature_dim: last dimension of input vectors
        out_feature_dim: last dimension of output vectors
        """
        super(FCBlock, self).__init__()
        self.fc_relu = nn.Sequential(
            nn.Linear(in_feature_dim, out_feature_dim),
            nn.BatchNorm1d(out_feature_dim),
            nn.Dropout2d(p=0.5), # 50%の割合でドロップアウトを実行
            # nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.fc_relu(x)
        return x

class LSTMBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        super().__init__()
        # LSTMの定義 batch_first=Trueで入力を(batch, seq_len, feature_dim)の形に
        # dropoutを指定するとLSTMの最終層以外の層の出力にドロップアウトが適用される
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout2d(p=dropout) # LSTMの最終層の出力に適用するドロップアウト
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """x: (batch_size, num_channels=8, freq_bins=257, time_frames=513)"""
        x = x.transpose(2, 3)
        """x: (batch_size, num_channles, time_frames=513, freq_bins=257)"""
        # チャンネルごとにLSTMで処理
        for i in range(x.shape[1]):
            each_x = x[:, i, :, :]
            """x: (batch_size, time_frames=513, freq_bins=257)"""
            each_x, _ = self.lstm(each_x)
            """each_x: (batch_size, time_frames=513, hidden_dim=256)"""
            # バッチノーマリゼーションを行うために次元入れ替え
            each_x = each_x.permute(0, 2, 1)
            """each_x: (batch_size, hidden_dim=256, time_frames=513)"""
            each_x = self.bn(each_x)
            each_x = each_x.permute(0, 2, 1)
            """each_x: (batch_size, time_frames=513, hidden_dim=256)"""
            each_x = self.dropout(each_x)
            each_x = each_x.unsqueeze(1)
            """each_x: (batch_size, num_channels=1, time_frames=513, hidden_dim=256)"""
            if i == 0:
                multi_x = each_x
            else:
                multi_x = torch.cat([multi_x, each_x], dim=1) # チャンネル方向に結合
        """multi_x: (batch_size, num_channels, time_frames=513, hidden_dim=256)"""
        # 全結合層に入力して出力のサイズを調整
        # 全結合層に入力するために変形
        multi_x = multi_x.contiguous().view(-1, multi_x.shape[3])
        """multi_x: (batch_size*num_channels*time_frames, hidden_dim=256)"""
        # 全結合層に入力
        multi_x = self.fc(multi_x)
        """multi_x: (batch_size*num_channels*time_frames, hidden_dim=257)"""
        # 変形
        x = multi_x.contiguous().view(x.shape[0], x.shape[1], x.shape[2], -1)
        """x: (batch_size, num_channels=8, time_frames=513, freq_bins=257)"""
        x = x.transpose(2, 3)
        """x: (batch_size, num_channels=8, freq_bins=257, time_frames=513)"""
        return x
        

# カーネルサイズ3×3のMulti-Channel Denoising Unetのモデルを定義
class UnetMaskEstimator_kernel3(nn.Module):
    def __init__(self):
        super(UnetMaskEstimator_kernel3, self).__init__()
        # 畳み込み層のパラメータを指定
        kernel_size = 3
        stride = 2
        padding = 1
        # encoderの層
        self.conv_block1 = ConvBlock(8, 16, kernel_size, stride, padding)
        self.conv_block2 = ConvBlock(16, 32, kernel_size, stride, padding)
        self.conv_block3 = ConvBlock(32, 64, kernel_size, stride, padding)
        self.conv_block4 = ConvBlock(64, 128, kernel_size, stride, padding)
        self.conv_block5 = ConvBlock(128, 256, kernel_size, stride, padding)
        self.conv_block6 = ConvBlock(256, 512, kernel_size, stride, padding)

        # decoderの層
        self.deconv_block1 = TransposeConvBlock(512, 256, kernel_size, stride, padding)
        self.deconv_block2 = TransposeConvBlock(512, 128, kernel_size, stride, padding)
        self.deconv_block3 = TransposeConvBlock(256, 64, kernel_size, stride, padding)
        self.deconv_block4 = TransposeConvBlock_without_dropout(128, 32, kernel_size, stride, padding)
        self.deconv_block5 = TransposeConvBlock_without_dropout(64, 16, kernel_size, stride, padding)
        # self.deconv_block6 = nn.Sequential(
        #     nn.ConvTranspose2d(32, 8, kernel_size, stride, padding),
        #     nn.Sigmoid()
        # )
        self.out_layer_target = nn.Sequential(
            nn.ConvTranspose2d(32, 8, kernel_size, stride, padding),
            nn.Sigmoid()
        )
        self.out_layer_noise = nn.Sequential(
            nn.ConvTranspose2d(32, 8, kernel_size, stride, padding),
            nn.Sigmoid()
        )
        # 活性化関数がsigmoidである転置畳み込み層の重みをxavierの重みで初期化
        init_weights(self.out_layer_target)
        init_weights(self.out_layer_noise)

    def forward(self, x):
        """x: (batch_size, channels=8, freq_bins=257, time_frames=513)"""
        # encoder forward
        h1 = self.conv_block1(x)
        h2 = self.conv_block2(h1)
        h3 = self.conv_block3(h2)
        h4 = self.conv_block4(h3)
        h5 = self.conv_block5(h4)
        h6 = self.conv_block6(h5)
        # decoder forward
        dh1 = self.deconv_block1(h6)
        dh2 = self.deconv_block2(torch.cat((dh1, h5), dim=1))
        dh3 = self.deconv_block3(torch.cat((dh2, h4), dim=1))
        dh4 = self.deconv_block4(torch.cat((dh3, h3), dim=1))
        dh5 = self.deconv_block5(torch.cat((dh4, h2), dim=1))
        # dh6 = self.deconv_block6(torch.cat((dh5, h1), dim=1))
        # output layer
        target_mask = self.out_layer_target(torch.cat((dh5, h1), dim=1))
        noise_mask = self.out_layer_noise(torch.cat((dh5, h1), dim=1))
        return target_mask, noise_mask

# カーネルサイズ3×3のMulti-Channel Denoising CNNのモデルを定義
class CNNMaskEstimator_kernel3(nn.Module):
    def __init__(self):
        super(CNNMaskEstimator_kernel3, self).__init__()
        # 畳み込み層のパラメータを指定
        kernel_size = 3
        stride = 2
        padding = 1
        # encoderの層
        self.conv_block1 = ConvBlock(8, 16, kernel_size, stride, padding)
        self.conv_block2 = ConvBlock(16, 32, kernel_size, stride, padding)
        self.conv_block3 = ConvBlock(32, 64, kernel_size, stride, padding)
        self.conv_block4 = ConvBlock(64, 128, kernel_size, stride, padding)
        self.conv_block5 = ConvBlock(128, 256, kernel_size, stride, padding)
        self.conv_block6 = ConvBlock(256, 512, kernel_size, stride, padding)

        # decoderの層
        self.deconv_block1 = TransposeConvBlock(512, 256, kernel_size, stride, padding)
        self.deconv_block2 = TransposeConvBlock(256, 128, kernel_size, stride, padding)
        self.deconv_block3 = TransposeConvBlock(128, 64, kernel_size, stride, padding)
        self.deconv_block4 = TransposeConvBlock_without_dropout(64, 32, kernel_size, stride, padding)
        self.deconv_block5 = TransposeConvBlock_without_dropout(32, 16, kernel_size, stride, padding)
        # self.deconv_block6 = nn.Sequential(
        #     nn.ConvTranspose2d(32, 8, kernel_size, stride, padding),
        #     nn.Sigmoid()
        # )
        self.out_layer_target = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size, stride, padding),
            nn.Sigmoid()
        )
        self.out_layer_noise = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size, stride, padding),
            nn.Sigmoid()
        )
        # 活性化関数がsigmoidである転置畳み込み層の重みをxavierの重みで初期化
        init_weights(self.out_layer_target)
        init_weights(self.out_layer_noise)

    def forward(self, x):
        """x: (batch_size, channels=8, freq_bins=257, time_steps=513)"""
        # encoder forward
        h1 = self.conv_block1(x)
        h2 = self.conv_block2(h1)
        h3 = self.conv_block3(h2)
        h4 = self.conv_block4(h3)
        h5 = self.conv_block5(h4)
        h6 = self.conv_block6(h5)
        # decoder forward
        dh1 = self.deconv_block1(h6)
#         dh2 = self.deconv_block2(torch.cat((dh1, h5), dim=1))
#         dh3 = self.deconv_block3(torch.cat((dh2, h4), dim=1))
#         dh4 = self.deconv_block4(torch.cat((dh3, h3), dim=1))
#         dh5 = self.deconv_block5(torch.cat((dh4, h2), dim=1))
        dh2 = self.deconv_block2(dh1)
        dh3 = self.deconv_block3(dh2)
        dh4 = self.deconv_block4(dh3)
        dh5 = self.deconv_block5(dh4)
        # dh6 = self.deconv_block6(torch.cat((dh5, h1), dim=1))
        # output layer
#         target_mask = self.out_layer_target(torch.cat((dh5, h1), dim=1))
#         noise_mask = self.out_layer_noise(torch.cat((dh5, h1), dim=1))
        target_mask = self.out_layer_target(dh5)
        noise_mask = self.out_layer_noise(dh5)
        return target_mask, noise_mask


# カーネルサイズ3×3のMulti-Channel Denoising Unetのモデルを定義
class UnetMaskEstimator_kernel3_single_mask(nn.Module):
    def __init__(self):
        super(UnetMaskEstimator_kernel3_single_mask, self).__init__()
        # 畳み込み層のパラメータを指定
        kernel_size = 3
        stride = 2
        padding = 1
        # encoderの層
        self.conv_block1 = ConvBlock(8, 16, kernel_size, stride, padding)
        self.conv_block2 = ConvBlock(16, 32, kernel_size, stride, padding)
        self.conv_block3 = ConvBlock(32, 64, kernel_size, stride, padding)
        self.conv_block4 = ConvBlock(64, 128, kernel_size, stride, padding)
        self.conv_block5 = ConvBlock(128, 256, kernel_size, stride, padding)
        self.conv_block6 = ConvBlock(256, 512, kernel_size, stride, padding)

        # decoderの層
        self.deconv_block1 = TransposeConvBlock(512, 256, kernel_size, stride, padding)
        self.deconv_block2 = TransposeConvBlock(512, 128, kernel_size, stride, padding)
        self.deconv_block3 = TransposeConvBlock(256, 64, kernel_size, stride, padding)
        self.deconv_block4 = TransposeConvBlock_without_dropout(128, 32, kernel_size, stride, padding)
        self.deconv_block5 = TransposeConvBlock_without_dropout(64, 16, kernel_size, stride, padding)
        # self.deconv_block6 = nn.Sequential(
        #     nn.ConvTranspose2d(32, 8, kernel_size, stride, padding),
        #     nn.Sigmoid()
        # )
        self.out_layer_target = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size, stride, padding),
            nn.Sigmoid()
        )
        self.out_layer_noise = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size, stride, padding),
            nn.Sigmoid()
        )
        # 活性化関数がsigmoidである転置畳み込み層の重みをxavierの重みで初期化
        init_weights(self.out_layer_target)
        init_weights(self.out_layer_noise)

    def forward(self, x):
        """x: (batch_size, channels=8, freq_bins=257, time_steps=513)"""
        # encoder forward
        h1 = self.conv_block1(x)
        h2 = self.conv_block2(h1)
        h3 = self.conv_block3(h2)
        h4 = self.conv_block4(h3)
        h5 = self.conv_block5(h4)
        h6 = self.conv_block6(h5)
        # decoder forward
        dh1 = self.deconv_block1(h6)
        dh2 = self.deconv_block2(torch.cat((dh1, h5), dim=1))
        dh3 = self.deconv_block3(torch.cat((dh2, h4), dim=1))
        dh4 = self.deconv_block4(torch.cat((dh3, h3), dim=1))
        dh5 = self.deconv_block5(torch.cat((dh4, h2), dim=1))
        # dh6 = self.deconv_block6(torch.cat((dh5, h1), dim=1))
        # output layer
        target_mask = self.out_layer_target(torch.cat((dh5, h1), dim=1))
        noise_mask = self.out_layer_noise(torch.cat((dh5, h1), dim=1))
        """target_mask: (batch_size, channels=1, freq_bins=257, time_steps=513)"""
        """noise_mask: (batch_size, channels=1, freq_bins=257, time_steps=513)"""
        return target_mask, noise_mask


# カーネルサイズ3×3のMulti-Channel Denoising Unetのモデルを定義
class UnetMaskEstimator_kernel3_single_mask_dereverb(nn.Module):
    def __init__(self):
        super(UnetMaskEstimator_kernel3_single_mask_dereverb, self).__init__()
        # 畳み込み層のパラメータを指定
        kernel_size = 3
        stride = 2
        padding = 1
        # LSTMのパラメータを指定
        num_freq_bins = 257
        hidden_dim = 256
        out_dim = 257
        num_layers = 1
        dropout = 0.5
        # encoderの層
        self.conv_block1 = ConvBlock(8, 16, kernel_size, stride, padding)
        self.conv_block2 = ConvBlock(16, 32, kernel_size, stride, padding)
        self.conv_block3 = ConvBlock(32, 64, kernel_size, stride, padding)
        self.conv_block4 = ConvBlock(64, 128, kernel_size, stride, padding)
        self.conv_block5 = ConvBlock(128, 256, kernel_size, stride, padding)
        self.conv_block6 = ConvBlock(256, 512, kernel_size, stride, padding)
        # decoderの層
        self.deconv_block1 = TransposeConvBlock(512, 256, kernel_size, stride, padding)
        self.deconv_block2 = TransposeConvBlock(512, 128, kernel_size, stride, padding)
        self.deconv_block3 = TransposeConvBlock(256, 64, kernel_size, stride, padding)
        self.deconv_block4 = TransposeConvBlock_without_dropout(128, 32, kernel_size, stride, padding)
        self.deconv_block5 = TransposeConvBlock_without_dropout(64, 16, kernel_size, stride, padding) 
        # 残響成分予測層および出力層
        self.speech_layer = nn.ConvTranspose2d(32, 1, kernel_size, stride, padding)
        self.noise_layer = nn.ConvTranspose2d(32, 1, kernel_size, stride, padding)
        self.speech_reverb_lstm_block = LSTMBlock(num_freq_bins, hidden_dim, out_dim, num_layers, dropout)
        self.noise_reverb_lstm_block = LSTMBlock(num_freq_bins, hidden_dim, out_dim, num_layers, dropout)
        self.activation_layer = nn.Sigmoid()
        # 活性化関数がsigmoidである全結合層、畳み込み層、転置畳み込み層の重みをxavierの重みで初期化
        init_weights(self.speech_layer)
        init_weights(self.noise_layer)
        init_weights(self.speech_reverb_lstm_block)
        init_weights(self.noise_reverb_lstm_block)

    def forward(self, x):
        """x: (batch_size, channels=8, freq_bins=257, time_steps=513)"""
        # encoder forward
        h1 = self.conv_block1(x)
        h2 = self.conv_block2(h1)
        h3 = self.conv_block3(h2)
        h4 = self.conv_block4(h3)
        h5 = self.conv_block5(h4)
        h6 = self.conv_block6(h5)
        # decoder forward
        dh1 = self.deconv_block1(h6)
        dh2 = self.deconv_block2(torch.cat((dh1, h5), dim=1))
        dh3 = self.deconv_block3(torch.cat((dh2, h4), dim=1))
        dh4 = self.deconv_block4(torch.cat((dh3, h3), dim=1))
        dh5 = self.deconv_block5(torch.cat((dh4, h2), dim=1))
        # dh6 = self.deconv_block6(torch.cat((dh5, h1), dim=1))
        # speech and noise branch 
        speech_feature = self.speech_layer(torch.cat((dh5, h1), dim=1))
        """speech_feature: (batch_size, num_channels, freq_bins=257, time_frames=513)"""
        noise_feature = self.noise_layer(torch.cat((dh5, h1), dim=1))
        """noise_feature: (batch_size, num_channels, freq_bins=257, time_frames=513)"""
        # reverb estimation layer
        speech_reverb = self.speech_reverb_lstm_block(speech_feature)
        """speech_reverb: (batch_size, num_channels=8, freq_bins=257, time_frames=513)"""
        noise_reverb = self.noise_reverb_lstm_block(noise_feature)
        """noise_reverb: (batch_size, num_channels=8, freq_bins=257, time_frames=513)"""
        # output layer
        speech_output = speech_feature - speech_reverb # test
        speech_mask = self.activation_layer(speech_output) 
        noise_output = noise_feature - noise_reverb # test
        noise_mask = self.activation_layer(noise_output)
        """speech_mask: (batch_size, channels=8, freq_bins=257, time_steps=513)"""
        """noise_mask: (batch_size, channels=8, freq_bins=257, time_steps=513)"""    
        return speech_mask, noise_mask


# カーネルサイズ3×3のMulti-Channel Denoising Unet two speakersのモデルを定義
class UnetMaskEstimator_kernel3_single_mask_two_speakers(nn.Module):
    def __init__(self):
        super(UnetMaskEstimator_kernel3_single_mask_two_speakers, self).__init__()
        # 畳み込み層のパラメータを指定
        kernel_size = 3
        stride = 2
        padding = 1
        # encoderの層
        self.conv_block1 = ConvBlock(8, 16, kernel_size, stride, padding)
        self.conv_block2 = ConvBlock(16, 32, kernel_size, stride, padding)
        self.conv_block3 = ConvBlock(32, 64, kernel_size, stride, padding)
        self.conv_block4 = ConvBlock(64, 128, kernel_size, stride, padding)
        self.conv_block5 = ConvBlock(128, 256, kernel_size, stride, padding)
        self.conv_block6 = ConvBlock(256, 512, kernel_size, stride, padding)

        # decoderの層
        self.deconv_block1 = TransposeConvBlock(512, 256, kernel_size, stride, padding)
        self.deconv_block2 = TransposeConvBlock(512, 128, kernel_size, stride, padding)
        self.deconv_block3 = TransposeConvBlock(256, 64, kernel_size, stride, padding)
        self.deconv_block4 = TransposeConvBlock_without_dropout(128, 32, kernel_size, stride, padding)
        self.deconv_block5 = TransposeConvBlock_without_dropout(64, 16, kernel_size, stride, padding)
        # self.deconv_block6 = nn.Sequential(
        #     nn.ConvTranspose2d(32, 8, kernel_size, stride, padding),
        #     nn.Sigmoid()
        # )
        self.out_layer_target = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size, stride, padding),
            nn.Sigmoid()
        )
        self.out_layer_interference = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size, stride, padding),
            nn.Sigmoid()
        )
        self.out_layer_noise = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size, stride, padding),
            nn.Sigmoid()
        )
        # 活性化関数がsigmoidである転置畳み込み層の重みをxavierの重みで初期化
        init_weights(self.out_layer_target)
        init_weights(self.out_layer_noise)

    def forward(self, x):
        """x: (batch_size, channels=8, freq_bins=257, time_steps=513)"""
        # encoder forward
        h1 = self.conv_block1(x)
        h2 = self.conv_block2(h1)
        h3 = self.conv_block3(h2)
        h4 = self.conv_block4(h3)
        h5 = self.conv_block5(h4)
        h6 = self.conv_block6(h5)
        # decoder forward
        dh1 = self.deconv_block1(h6)
        dh2 = self.deconv_block2(torch.cat((dh1, h5), dim=1))
        dh3 = self.deconv_block3(torch.cat((dh2, h4), dim=1))
        dh4 = self.deconv_block4(torch.cat((dh3, h3), dim=1))
        dh5 = self.deconv_block5(torch.cat((dh4, h2), dim=1))
        # dh6 = self.deconv_block6(torch.cat((dh5, h1), dim=1))
        # output layer
        target_mask = self.out_layer_target(torch.cat((dh5, h1), dim=1))
        interference_mask = self.out_layer_interference(torch.cat((dh5, h1), dim=1))
        noise_mask = self.out_layer_noise(torch.cat((dh5, h1), dim=1))
        """target_mask: (batch_size, channels=1, freq_bins=257, time_steps=513)"""
        """interference_mask: (batch_size, channels=1, freq_bins=257, time_steps=513)"""
        """noise_mask: (batch_size, channels=1, freq_bins=257, time_steps=513)"""
        return target_mask, interference_mask, noise_mask

# 全結合層を並べたマスク推定モデルを定義
class FCMaskEstimator(nn.Module):
    def __init__(self):
        super(FCMaskEstimator, self).__init__()
 
        self.fc_block1 = FCBlock(257, 512)
        self.fc_block2 = FCBlock(512, 512)
        self.fc_block3 = FCBlock(512, 512)
        self.fc_block4 = FCBlock(512, 512)
        self.fc_block5 = FCBlock(512, 257)

        self.out_layer_target = nn.Sequential(
            nn.Linear(257, 257),
            nn.Sigmoid()
        )
        self.out_layer_noise = nn.Sequential(
            nn.Linear(257, 257),
            nn.Sigmoid()
        )
        # 活性化関数がsigmoidである全結合層の重みをxavierの重みで初期化
        init_weights(self.out_layer_target)
        init_weights(self.out_layer_noise)

    def forward(self, x):
        """x: (batch_size, num_channels=8, freq_bins=257, time_frames=513)"""
        input_x = x.transpose(2, 3)
        """input_x: (batch_size, num_channels=8, time_frames=513, freq_bins=257)"""
        x = input_x.contiguous().view(input_x.shape[0] * input_x.shape[1] * input_x.shape[2], -1)
        """x: (batch_size*num_channels*time_frames, freq_bins=257)"""
        x = self.fc_block1(x)
        x = self.fc_block2(x)
        x = self.fc_block3(x)
        x = self.fc_block4(x)
        x = self.fc_block5(x)
        """x: (batch_size*num_channels*time_frames, freq_bins=257)"""
        x = x.contiguous().view(input_x.shape[0], input_x.shape[1], input_x.shape[2], -1)
        """x: (batch_size, num_channels=8, time_frames=513, freq_bins=257)"""
        # 出力層（分岐）
        target_mask = self.out_layer_target(x)
        """target_mask: (batch_size, num_channels=8, time_frames=513, freq_bins=257)"""
        noise_mask = self.out_layer_noise(x)
        """noise_mask: (batch_size, num_channels=8, time_frames=513, freq_bins=257)"""
        target_mask = target_mask.transpose(2, 3)
        """target_mask: (batch_size, num_channels=8, freq_bins=257, time_frames=513)"""
        noise_mask = noise_mask.transpose(2, 3)
        """noise_mask: (batch_size, num_channels=8, freq_bins=257, time_frames=513)"""
        return target_mask, noise_mask

# BLSTMを並べたマスク推定モデルを定義
class BLSTMMaskEstimator(nn.Module):
    def __init__(self):
        super(BLSTMMaskEstimator, self).__init__()
        num_freq_bins = 257
 
        # BLSTMの定義 batch_first=Trueで入力を(batch, seq_len, feature_dim)の形に
        self.blstm = nn.LSTM(num_freq_bins, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout2d(p=0.5)

        self.fc_block1 = FCBlock(512, num_freq_bins)
        self.fc_block2 = FCBlock(num_freq_bins, num_freq_bins)
        # 出力層の定義
        self.out_layer_target = nn.Sequential(
            nn.Linear(num_freq_bins, num_freq_bins),
            nn.Sigmoid()
        )
        self.out_layer_noise = nn.Sequential(
            nn.Linear(num_freq_bins, num_freq_bins),
            nn.Sigmoid()
        )
        # 活性化関数がsigmoidである全結合層の重みをxavierの重みで初期化
        init_weights(self.out_layer_target)
        init_weights(self.out_layer_noise)

    def forward(self, x):
        """x: (batch_size, num_channels=8, freq_bins=257, time_frames=513)"""
        x = x.transpose(2, 3)
        """x: (batch_size, num_channels=8, time_frames=513, freq_bins=257)"""
        # チャンネルごとにLSTMで処理
        for i in range(x.shape[1]):
            each_x = x[:, i, :, :]
            """x: (batch_size, time_frames=513, freq_bins=257)"""
            each_x, _ = self.blstm(each_x)
            """x: (batch_size, time_frames=513, hidden_dim=256*2)"""
            # バッチノーマリゼーションを行うために次元入れ替え
            each_x = each_x.permute(0, 2, 1)
            """each_x: (batch_size, hidden_dim=512, time_frames=513)"""
            each_x = self.bn(each_x)
            each_x = each_x.permute(0, 2, 1)
            """each_x: (batch_size, time_frames=513, hidden_dim=512)"""
            each_x = self.dropout(each_x)
            each_x = each_x.unsqueeze(1)
            """x: (batch_size, num_channels=1, time_frames=513, hidden_dim=512)"""
            if i == 0:
                multi_x = each_x
            else:
                multi_x = torch.cat([multi_x, each_x], dim=1) # チャンネル方向に結合
        """multi_x: (batch_size, num_channels=8, time_frames=513, hidden_dim=512)"""
        # バッチノーマリゼーションを行うために次元調整
        x = multi_x.contiguous().view(-1, multi_x.shape[3])
        """x: (batch_size*num_channels*time_frames, hidden_dim=512)"""
        x = self.fc_block1(x)
        """x: (batch_size*num_channels*time_frames, hidden_dim=257)"""
        x = self.fc_block2(x)
        """x: (batch_size*num_channels*time_frames, hidden_dim=257)"""
        x = x.contiguous().view(multi_x.shape[0], multi_x.shape[1], multi_x.shape[2], -1)
        """x: (batch_size, num_channels=8, time_frames=513, freq_bins=257)"""
        # 出力層（分岐）
        target_mask = self.out_layer_target(x)
        """target_mask: (batch_size, num_channels=8, time_frames=513, freq_bins=257)"""
        noise_mask = self.out_layer_noise(x)
        """noise_mask: (batch_size, num_channels=8, time_frames=513, freq_bins=257)"""
        target_mask = target_mask.transpose(2, 3)
        """target_mask: (batch_size, num_channels=8, freq_bins=257, time_frames=513)"""
        noise_mask = noise_mask.transpose(2, 3)
        """noise_mask: (batch_size, num_channels=8, freq_bins=257, time_frames=513)"""
        return target_mask, noise_mask

# 2層のBLSTMを並べたマスク推定モデルを定義
class BLSTMMaskEstimator2(nn.Module):
    def __init__(self):
        super(BLSTMMaskEstimator2, self).__init__()
        num_freq_bins = 257
 
        # BLSTMの定義 batch_first=Trueで入力を(batch, seq_len, feature_dim)の形に
        self.blstm1 = nn.LSTM(num_freq_bins, 256, batch_first=True, bidirectional=True)
        self.blstm2 = nn.LSTM(512, 256, batch_first=True, bidirectional=True)
        
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout2d(p=0.5)

        self.fc_block1 = FCBlock(512, num_freq_bins)
        self.fc_block2 = FCBlock(num_freq_bins, num_freq_bins)
        # 出力層の定義
        self.out_layer_target = nn.Sequential(
            nn.Linear(num_freq_bins, num_freq_bins),
            nn.Sigmoid()
        )
        self.out_layer_noise = nn.Sequential(
            nn.Linear(num_freq_bins, num_freq_bins),
            nn.Sigmoid()
        )
        # 活性化関数がsigmoidである全結合層の重みをxavierの重みで初期化
        init_weights(self.out_layer_target)
        init_weights(self.out_layer_noise)

    def forward(self, x):
        """x: (batch_size, num_channels=8, freq_bins=257, time_frames=513)"""
        x = x.transpose(2, 3)
        """x: (batch_size, num_channels=8, time_frames=513, freq_bins=257)"""
        # チャンネルごとにLSTMで処理
        for i in range(x.shape[1]):
            each_x = x[:, i, :, :]
            """x: (batch_size, time_frames=513, freq_bins=257)"""
            each_x, _ = self.blstm1(each_x)
            """each_x: (batch_size, time_frames=513, hidden_dim=256*2)"""
            each_x, _ = self.blstm2(each_x)
            """each_x: (batch_size, time_frames=513, hidden_dim=256*2)"""
            # バッチノーマリゼーションを行うために次元入れ替え
            each_x = each_x.permute(0, 2, 1)
            """each_x: (batch_size, hidden_dim=512, time_frames=513)"""
            each_x = self.bn(each_x)
            each_x = each_x.permute(0, 2, 1)
            """each_x: (batch_size, time_frames=513, hidden_dim=512)"""
            each_x = self.dropout(each_x)
            each_x = each_x.unsqueeze(1)
            """x: (batch_size, num_channels=1, time_frames=513, hidden_dim=512)"""
            if i == 0:
                multi_x = each_x
            else:
                multi_x = torch.cat([multi_x, each_x], dim=1) # チャンネル方向に結合
        """multi_x: (batch_size, num_channels=8, time_frames=513, hidden_dim=512)"""
        # バッチノーマリゼーションを行うために次元調整
        x = multi_x.contiguous().view(-1, multi_x.shape[3])
        """x: (batch_size*num_channels*time_frames, hidden_dim=512)"""
        x = self.fc_block1(x)
        """x: (batch_size*num_channels*time_frames, hidden_dim=257)"""
        x = self.fc_block2(x)
        """x: (batch_size*num_channels*time_frames, hidden_dim=257)"""
        x = x.contiguous().view(multi_x.shape[0], multi_x.shape[1], multi_x.shape[2], -1)
        """x: (batch_size, num_channels=8, time_frames=513, freq_bins=257)"""
        # 出力層（分岐）
        target_mask = self.out_layer_target(x)
        """target_mask: (batch_size, num_channels=8, time_frames=513, freq_bins=257)"""
        noise_mask = self.out_layer_noise(x)
        """noise_mask: (batch_size, num_channels=8, time_frames=513, freq_bins=257)"""
        target_mask = target_mask.transpose(2, 3)
        """target_mask: (batch_size, num_channels=8, freq_bins=257, time_frames=513)"""
        noise_mask = noise_mask.transpose(2, 3)
        """noise_mask: (batch_size, num_channels=8, freq_bins=257, time_frames=513)"""
        return target_mask, noise_mask

#######################################Multi-channel Complex Unet#########################################
# 複素畳み込み層
class CConv2d(nn.Module):
    """
    Class of complex valued convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        self.real_conv = nn.Conv2d(in_channels=self.in_channels, 
                                   out_channels=self.out_channels, 
                                   kernel_size=self.kernel_size, 
                                   padding=self.padding, 
                                   stride=self.stride)
        
        self.im_conv = nn.Conv2d(in_channels=self.in_channels, 
                                 out_channels=self.out_channels, 
                                 kernel_size=self.kernel_size, 
                                 padding=self.padding, 
                                 stride=self.stride)
        
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)
        
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)
        
        output = torch.stack([c_real, c_im], dim=-1)
        return output

# 複素転置畳み込み層
class CConvTranspose2d(nn.Module):
    """
      Class of complex valued dilation convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()
        
        self.in_channels = in_channels

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride
        
        self.real_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding,
                                            padding=self.padding,
                                            stride=self.stride)
        
        self.im_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding, 
                                            padding=self.padding,
                                            stride=self.stride)
        
        
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)
        
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)
        
        output = torch.stack([ct_real, ct_im], dim=-1)
        return output

# 複素バッチノーマリゼーション層
class CBatchNorm2d(nn.Module):
    """
    Class of complex valued batch normalization layer
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                      affine=self.affine, track_running_stats=self.track_running_stats)
        self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                    affine=self.affine, track_running_stats=self.track_running_stats) 
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        n_real = self.real_b(x_real)
        n_im = self.im_b(x_im)  
        
        output = torch.stack([n_real, n_im], dim=-1)
        return output

class Encoder(nn.Module):
    """
    Class of downsample block
    """
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45, padding=(0,0)):
        super().__init__()
        
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cconv = CConv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        
        conved = self.cconv(x)
        normed = self.cbn(conved)
        acted = self.leaky_relu(normed)
        
        return acted

class Decoder(nn.Module):
    """
    Class of upsample block
    """
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45,
                 output_padding=(0,0), padding=(0,0), last_layer=False):
        super().__init__()
        
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding
        
        self.last_layer = last_layer
        
        self.cconvt = CConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, output_padding=self.output_padding, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        
        conved = self.cconvt(x)
        
        if not self.last_layer:
            normed = self.cbn(conved)
            output = self.leaky_relu(normed)
        else:
            m_phase = conved / (torch.abs(conved) + 1e-8)
            m_mag = torch.tanh(torch.abs(conved))
            output = m_phase * m_mag
            
        return output

# Multi-channel Complex Unetモデルの定義
class MCComplexUnet(nn.Module):
    """
    Deep Complex U-Net class of the model.
    """
    def __init__(self):
        super().__init__()
        
        # チャンネルほぼ一定バージョン
        # downsampling/encoding
        self.downsample0 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=8, out_channels=45, padding=(1,1))
        self.downsample1 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=45, out_channels=90, padding=(1,1))
        self.downsample2 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, out_channels=90, padding=(1,1))
        self.downsample3 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, out_channels=90, padding=(1,1))
        self.downsample4 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, out_channels=90, padding=(1,1))
        self.downsample5 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, out_channels=90, padding=(1,1))
        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, out_channels=90, padding=(1,1))
        self.upsample1 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=180, out_channels=90, padding=(1,1))
        self.upsample2 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=180, out_channels=90, padding=(1,1))
        self.upsample3 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=180, out_channels=90, padding=(1,1))
        self.upsample4 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=180, out_channels=45, padding=(1,1))
        # output (speech and noise)
        self.out_layer_speech = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, out_channels=8, padding=(1,1), last_layer=True)
        self.out_layer_noise = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, out_channels=8, padding=(1,1), last_layer=True)
#         # output (speech only)
#         self.out_layer_speech = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, out_channels=8, padding=(1,1), last_layer=True)

        # # チャンネル数大きく変化
        # # downsampling/encoding
        # self.downsample0 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=8, out_channels=16, padding=(1,1))
        # self.downsample1 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=16, out_channels=32, padding=(1,1))
        # self.downsample2 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=32, out_channels=64, padding=(1,1))
        # self.downsample3 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, out_channels=128, padding=(1,1))
        # self.downsample4 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=256, padding=(1,1))
        # self.downsample5 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=256, out_channels=512, padding=(1,1))
        # # upsampling/decoding
        # self.upsample0 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=512, out_channels=256, padding=(1,1))
        # self.upsample1 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=512, out_channels=128, padding=(1,1))
        # self.upsample2 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=256, out_channels=64, padding=(1,1))
        # self.upsample3 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=32, padding=(1,1))
        # self.upsample4 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, out_channels=16, padding=(1,1))
        # # # output (speech only)
        # # self.out_layer_speech = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=32, out_channels=8, padding=(1,1), last_layer=True)
        # # output (speech and noise)
        # self.out_layer_speech = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=32, out_channels=8, padding=(1,1), last_layer=True)
        # self.out_layer_noise = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=32, out_channels=8, padding=(1,1), last_layer=True)
        
    def forward(self, x, is_istft=False):
        """x: (batch_size, num_channels, freq_bins, time_steps, real-imaginary)"""
        # downsampling/encoding
        d0 = self.downsample0(x)
        d1 = self.downsample1(d0)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2) 
        d4 = self.downsample4(d3)
        d5 = self.downsample5(d4) 
        # upsampling/decoding 
        u0 = self.upsample0(d5)
        u1 = self.upsample1(torch.cat((u0, d4), dim=1))        
        u2 = self.upsample2(torch.cat((u1, d3), dim=1))        
        u3 = self.upsample3(torch.cat((u2, d2), dim=1))
        u4 = self.upsample4(torch.cat((u3, d1), dim=1))
        c4 = torch.cat((u4, d0), dim=1)
        
#         # output (speech only)
#         speech_mask = self.out_layer_speech(c4) # estimate complex ideal ratio mask (cIRM)
#         speech_output = speech_mask * x
#         return speech_output
        
        # output (speech and noise)
        speech_mask = self.out_layer_speech(c4) # estimate complex ideal ratio mask (cIRM)
        noise_mask = self.out_layer_noise(c4) # estimate complex ideal ratio mask (cIRM)
        speech_output = speech_mask * x
        noise_output = noise_mask * x
        return speech_output, noise_output

#######################################Multi-channel Conv-Tasnet#########################################
class ConvBlock1D(torch.nn.Module):
    """1D Convolutional block.

    Args:
        io_channels (int): The number of input/output channels, <B, Sc>
        hidden_channels (int): The number of channels in the internal layers, <H>.
        kernel_size (int): The convolution kernel size of the middle layer, <P>.
        padding (int): Padding value of the convolution in the middle layer.
        dilation (int): Dilation value of the convolution in the middle layer.
        no_redisual (bool): Disable residual block/output.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.
    """

    def __init__(
        self,
        io_channels: int,
        hidden_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int = 1,
        no_residual: bool = False,
    ):
        super().__init__()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=io_channels, out_channels=hidden_channels, kernel_size=1
            ),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
            torch.nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_channels,
            ),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
        )

        self.res_out = (
            None
            if no_residual
            else torch.nn.Conv1d(
                in_channels=hidden_channels, out_channels=io_channels, kernel_size=1
            )
        )
        self.skip_out = torch.nn.Conv1d(
            in_channels=hidden_channels, out_channels=io_channels, kernel_size=1
        )

    def forward(self, input: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        feature = self.conv_layers(input)
        if self.res_out is None:
            residual = None
        else:
            residual = self.res_out(feature)
        skip_out = self.skip_out(feature)
        return residual, skip_out


class MaskGenerator(torch.nn.Module):
    """TCN (Temporal Convolution Network) Separation Module

    Generates masks for separation.

    Args:
        input_dim (int): Input feature dimension, <N>.
        num_sources (int): The number of sources to separate.
        kernel_size (int): The convolution kernel size of conv blocks, <P>.
        num_featrs (int): Input/output feature dimenstion of conv blocks, <B, Sc>.
        num_hidden (int): Intermediate feature dimention of conv blocks, <H>
        num_layers (int): The number of conv blocks in one stack, <X>.
        num_stacks (int): The number of conv block stacks, <R>.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.
    """

    def __init__(
        self,
        input_dim: int,
        num_sources: int,
        kernel_size: int,
        num_feats: int,
        num_hidden: int,
        num_layers: int,
        num_stacks: int,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_sources = num_sources

        self.input_norm = torch.nn.GroupNorm(
            num_groups=1, num_channels=input_dim, eps=1e-8
        )
        self.input_conv = torch.nn.Conv1d(
            in_channels=input_dim, out_channels=num_feats, kernel_size=1
        )

        self.receptive_field = 0
        self.conv_layers = torch.nn.ModuleList([])
        for s in range(num_stacks):
            for l in range(num_layers):
                multi = 2 ** l
                self.conv_layers.append(
                    ConvBlock1D(
                        io_channels=num_feats,
                        hidden_channels=num_hidden,
                        kernel_size=kernel_size,
                        dilation=multi,
                        padding=multi,
                        # The last ConvBlock does not need residual
                        no_residual=(l == (num_layers - 1) and s == (num_stacks - 1)),
                    )
                )
                self.receptive_field += (
                    kernel_size if s == 0 and l == 0 else (kernel_size - 1) * multi
                )
        self.output_prelu = torch.nn.PReLU()
        self.output_conv = torch.nn.Conv1d(
            in_channels=num_feats, out_channels=input_dim * num_sources, kernel_size=1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Generate separation mask.

        Args:
            input (torch.Tensor): 3D Tensor with shape [batch, features, frames]

        Returns:
            torch.Tensor: shape [batch, num_sources, features, frames]
        """
        batch_size = input.shape[0]
        feats = self.input_norm(input)
        feats = self.input_conv(feats)
        output = 0.0
        for layer in self.conv_layers:
            residual, skip = layer(feats)
            if residual is not None:  # the last conv layer does not produce residual
                feats = feats + residual
            output = output + skip
        output = self.output_prelu(output)
        output = self.output_conv(output)
        output = torch.sigmoid(output)
        return output.view(batch_size, self.num_sources, self.input_dim, -1)


class MCConvTasNet(torch.nn.Module):
    """Conv-TasNet: a fully-convolutional time-domain audio separation network :footcite:`Luo_2019`.

    Args:
        num_sources (int): The number of sources to split.
        enc_kernel_size (int): The convolution kernel size of the encoder/decoder, <L>.
        enc_num_feats (int): The feature dimensions passed to mask generator, <N>.
        msk_kernel_size (int): The convolution kernel size of the mask generator, <P>.
        msk_num_feats (int): The input/output feature dimension of conv block in the mask generator, <B, Sc>.
        msk_num_hidden_feats (int): The internal feature dimension of conv block of the mask generator, <H>.
        msk_num_layers (int): The number of layers in one conv block of the mask generator, <X>.
        msk_num_stacks (int): The numbr of conv blocks of the mask generator, <R>.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.
    """

    def __init__(
        self,
        num_sources: int = 2,
        # encoder/decoder parameters
        enc_kernel_size: int = 16,
        enc_num_feats: int = 512,
        # mask generator parameters
        msk_kernel_size: int = 3,
        msk_num_feats: int = 128,
        msk_num_hidden_feats: int = 512,
        msk_num_layers: int = 8,
        msk_num_stacks: int = 3,
    ):
        super().__init__()

        self.num_sources = num_sources
        self.enc_num_feats = enc_num_feats
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2

        self.encoder = torch.nn.Conv1d(
            # in_channels=1,
            in_channels=8,
            out_channels=enc_num_feats,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )
        self.mask_generator = MaskGenerator(
            input_dim=enc_num_feats,
            num_sources=num_sources,
            kernel_size=msk_kernel_size,
            num_feats=msk_num_feats,
            num_hidden=msk_num_hidden_feats,
            num_layers=msk_num_layers,
            num_stacks=msk_num_stacks,
        )
        self.decoder = torch.nn.ConvTranspose1d(
            in_channels=enc_num_feats,
            # out_channels=1,
            out_channels=8,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )

    def _align_num_frames_with_strides(self, input: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pad input Tensor so that the end of the input tensor corresponds with

        1. (if kernel size is odd) the center of the last convolution kernel
        or 2. (if kernel size is even) the end of the first half of the last convolution kernel

        Assumption:
            The resulting Tensor will be padded with the size of stride (== kernel_width // 2)
            on the both ends in Conv1D

        |<--- k_1 --->|
        |      |            |<-- k_n-1 -->|
        |      |                  |  |<--- k_n --->|
        |      |                  |         |      |
        |      |                  |         |      |
        |      v                  v         v      |
        |<---->|<--- input signal --->|<--->|<---->|
         stride                         PAD  stride

        Args:
            input (torch.Tensor): 3D Tensor with shape (batch_size, channels==1, frames)

        Returns:
            torch.Tensor: Padded Tensor
            int: Number of paddings performed
        """
        batch_size, num_channels, num_frames = input.shape
        is_odd = self.enc_kernel_size % 2
        num_strides = (num_frames - is_odd) // self.enc_stride
        num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
        if num_remainings == 0:
            return input, 0

        num_paddings = self.enc_stride - num_remainings
        pad = torch.zeros(
            batch_size,
            num_channels,
            num_paddings,
            dtype=input.dtype,
            device=input.device,
        )
        return torch.cat([input, pad], 2), num_paddings

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Perform source separation. Generate audio source waveforms.

        Args:
            input (torch.Tensor): 3D Tensor with shape [batch, channel==1, frames]

        Returns:
            torch.Tensor: 3D Tensor with shape [batch, channel==num_sources, frames]
        """
        # if input.ndim != 3 or input.shape[1] != 1:
        #     raise ValueError(
        #         f"Expected 3D tensor (batch, channel==1, frames). Found: {input.shape}"
        #     )

        # B: batch size
        # L: input frame length
        # L': padded input frame length
        # F: feature dimension
        # M: feature frame length
        # S: number of sources
        # C: number of channels （新たに追加）

        padded, num_pads = self._align_num_frames_with_strides(input)  # B, 8, L'
        """padded: (batch_size, num_channels, num_samples)"""
        batch_size, num_channels, num_padded_frames = padded.shape[0], padded.shape[1], padded.shape[2]
        feats = self.encoder(padded)  # B, F, M
        """feats: (batch_size, feature_dim, feature_frame_length)"""
        masked = self.mask_generator(feats) * feats.unsqueeze(1)  # B, S, F, M
        masked = masked.view(
            batch_size * self.num_sources, self.enc_num_feats, -1
        )  # B*S, F, M
        decoded = self.decoder(masked)  # B*S, 1, L'
        output = decoded.view(
            # batch_size, self.num_sources, num_padded_frames # B, S, L'
            batch_size, self.num_sources, num_channels, num_padded_frames # B, S, C, L'
        )
        if num_pads > 0:
            output = output[..., :-num_pads]  # B, S, C, L
        return output
####################################################################################################



def main():
    # # FC or BLSTM
    # model = BLSTMMaskEstimator()
    # tensor = torch.rand(8, 8, 257, 301) # (batch_size, num_channels, freq_bins, time_frames)
    # output1, output2 = model(tensor)
    # print(output1.shape)
    # print(output2.shape)

    # # MaskU-Net
    # model = UnetMaskEstimator_kernel3_single_mask()
    # tensor = torch.rand(8, 8, 257, 513) # (batch_size, num_channels, freq_bins, time_frames)
    # output1, output2 = model(tensor)
    # print(output1.shape)
    # print(output2.shape)

    # MCComplexUnet
    SAMPLE_RATE = 16000
    N_FFT = 512
    HOP_LENGTH = 160
    mccunet = MCComplexUnet()
    total_params = sum(p.numel() for p in mccunet.parameters())
    print("total params:",total_params)
    # # start = time.perf_counter()
    # # random input
    # # random_input = torch.randn(1, 8, 257, 513, 2)
    # # print("input_shape:", random_input.shape)
    # # file input
    # file = "./test/p232_414_p257_074_noise_mix/p232_414_p257_074_mixed.wav" # 8ch
    # # file = "./test/p232_414_p257_074_mix_single_channel/16kHz/p232_414_p257_074_noise_mixed_single_channel.wav" # 1ch
    # waveform, _ = torchaudio.load(file)
    # """waveform: (num_channels, num_samples)"""
    # x_noisy_stft = torch.stft(input=waveform, n_fft=512, hop_length=160, normalized=True, return_complex=False)
    # # x_noisy_stft = torch.stft(input=waveform, n_fft=512, hop_length=160, normalized=False, return_complex=False)
    # """x_noisy_stft: (num_channels, freq_bins, time_steps, real-imaginary)"""
    # x_noisy_stft = nn.ZeroPad2d((0, 0, 0, 513-x_noisy_stft.shape[2]))(x_noisy_stft) # padding
    # """x_noisy_stft: (num_channels, freq_bins, time_steps=513, real-imaginary)"""
    # x_noisy_stft = torch.unsqueeze(x_noisy_stft, dim=0)
    # """x_noisy_stft: (batch_size, num_channels, freq_bins, time_steps, real-imaginary)"""
    # speech_output, noise_output = mccunet(x_noisy_stft)
    # """speech_output: (batch_size, num_samples), noise_output: (batch_size, num_samples)"""
    # # end = time.perf_counter()
    # # print("処理時間：", end-start)
    # print("speech_output:", speech_output.shape)
    # print("noise_output:", speech_output.shape)

    # # Multi-channel Conv-TasNet
    # SAMPLE_RATE = 16000
    # N_FFT = 512
    # HOP_LENGTH = 160
    # mc_conv_tasnet = MCConvTasNet()
    # total_params = sum(p.numel() for p in mc_conv_tasnet.parameters())
    # print("total params:",total_params)
    # # start = time.perf_counter()
    # random_input = torch.randn(1, 8, 48000)
    # """random_input: (batch_size, num_channels, num_samples)"""
    # output = mc_conv_tasnet(random_input)
    # """output: (batch_size, num_speakers, num_channels, num_samples)"""
    # # end = time.perf_counter()
    # # print("処理時間：", end-start)
    # print("output:", output.shape)


    # summary(model, (1, 257, 513))


if __name__ == "__main__":
    main()
