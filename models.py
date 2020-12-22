import torch
import torch.nn as nn
# from torchsummary import summary

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
        return target_mask, noise_mask

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
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.fc_relu(x)
        return x

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
        """x: (batch_size, num_channels=8, freq_bins=257, time_steps=513)"""
        input_x = x.transpose(2, 3)
        """input_x: (batch_size, num_channels=8, time_steps=513, freq_bins=257)"""
        x = input_x.contiguous().view(input_x.shape[0] * input_x.shape[1] * input_x.shape[2], -1)
        """x: (batch_size*num_channels*time_steps, freq_bins=257)"""
        x = self.fc_block1(x)
        x = self.fc_block2(x)
        x = self.fc_block3(x)
        x = self.fc_block4(x)
        x = self.fc_block5(x)
        """x: (batch_size*num_channels*time_steps, freq_bins=257)"""
        x = x.contiguous().view(input_x.shape[0], input_x.shape[1], input_x.shape[2], -1)
        """x: (batch_size, num_channels=8, time_steps=513, freq_bins=257)"""
        # 出力層（分岐）
        target_mask = self.out_layer_target(x)
        """target_mask: (batch_size, num_channels=8, time_steps=513, freq_bins=257)"""
        noise_mask = self.out_layer_noise(x)
        """noise_mask: (batch_size, num_channels=8, time_steps=513, freq_bins=257)"""
        target_mask = target_mask.transpose(2, 3)
        """target_mask: (batch_size, num_channels=8, freq_bins=257, time_steps=513)"""
        noise_mask = noise_mask.transpose(2, 3)
        """noise_mask: (batch_size, num_channels=8, freq_bins=257, time_steps=513)"""
        return target_mask, noise_mask

# BLSTMを並べたマスク推定モデルを定義
class BLSTMMaskEstimator(nn.Module):
    def __init__(self):
        super(BLSTMMaskEstimator, self).__init__()
        num_freq_bins = 257
 
        # BLSTMの定義 batch_first=Trueで入力を(batch, seq_len, feature_dim)の形に
        self.blstm = nn.LSTM(num_freq_bins, 256, batch_first=True, bidirectional=True)
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
        """x: (batch_size, num_channels=8, freq_bins=257, time_steps=513)"""
        x = x.transpose(2, 3)
        """x: (batch_size, num_channels=8, time_steps=513, freq_bins=257)"""
        # チャンネルごとにLSTMで処理
        for i in range(x.shape[1]):
            each_x = x[:, i, :, :]
            """x: (batch_size, time_steps=513, freq_bins=257)"""
            each_x, _ = self.blstm(each_x)
            """x: (batch_size, time_steps=513, hidden_dim=256*2)"""
            # バッチノーマリゼーションを行うために次元入れ替え
            each_x = each_x.permute(0, 2, 1)
            """each_x: (batch_size, hidden_dim=512, time_steps=513)"""
            each_x = self.bn(each_x)
            each_x = each_x.permute(0, 2, 1)
            """each_x: (batch_size, time_steps=513, hidden_dim=512)"""
            each_x = self.dropout(each_x)
            each_x = each_x.unsqueeze(1)
            """x: (batch_size, num_channels=1, time_steps=513, hidden_dim=512)"""
            if i == 0:
                multi_x = each_x
            else:
                multi_x = torch.cat([multi_x, each_x], dim=1) # チャンネル方向に結合
        """multi_x: (batch_size, num_channels=8, time_steps=513, hidden_dim=512)"""
        # バッチノーマリゼーションを行うために次元調整
        x = multi_x.contiguous().view(-1, multi_x.shape[3])
        """x: (batch_size*num_channels*time_steps, hidden_dim=512)"""
        x = self.fc_block1(x)
        """x: (batch_size*num_channels*time_steps, hidden_dim=257)"""
        x = self.fc_block2(x)
        """x: (batch_size*num_channels*time_steps, hidden_dim=257)"""
        x = x.contiguous().view(multi_x.shape[0], multi_x.shape[1], multi_x.shape[2], -1)
        """x: (batch_size, num_channels=8, time_steps=513, freq_bins=257)"""
        # 出力層（分岐）
        target_mask = self.out_layer_target(x)
        """target_mask: (batch_size, num_channels=8, time_steps=513, freq_bins=257)"""
        noise_mask = self.out_layer_noise(x)
        """noise_mask: (batch_size, num_channels=8, time_steps=513, freq_bins=257)"""
        target_mask = target_mask.transpose(2, 3)
        """target_mask: (batch_size, num_channels=8, freq_bins=257, time_steps=513)"""
        noise_mask = noise_mask.transpose(2, 3)
        """noise_mask: (batch_size, num_channels=8, freq_bins=257, time_steps=513)"""
        return target_mask, noise_mask

def main():
    # # FC or BLSTM
    # model = BLSTMMaskEstimator()
    # tensor = torch.rand(8, 8, 257, 301) # (batch_size, num_channels, freq_bins, time_steps)
    # output1, output2 = model(tensor)
    # print(output1.shape)
    # print(output2.shape)

    # MaskU-Net
    model = UnetMaskEstimator_kernel3()
    tensor = torch.rand(8, 8, 257, 513) # (batch_size, num_channels, freq_bins, time_steps)
    output1, output2 = model(tensor)
    print(output1.shape)
    print(output2.shape)


    # summary(model, (1, 257, 513))


if __name__ == "__main__":
    main()
