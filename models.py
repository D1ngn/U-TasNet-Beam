import torch
import torch.nn as nn
# from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding), # ２次元データの畳み込み演算を行う層　引数は入力のチャンネル数、出力のチャンネル数、カーネル(フィルタ)の大きさ
            nn.BatchNorm2d(out_channels), # batch normalizationを行う層　引数は入力データのチャンネル数
            nn.LeakyReLU(0.2, inplace=True) # inplace=Trueにすることで、使用メモリを削減
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class TransposeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(TransposeConvBlock, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding), # ２次元データの畳み込み演算を行う層　引数は入力のチャンネル数、出力のチャンネル数、カーネル(フィルタ)の大きさ
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

# Unetのモデルを定義
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        # encoderの層
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1), # ２次元データの畳み込み演算を行う層　引数は入力のチャンネル数、出力のチャンネル数、カーネル(フィルタ)の大きさ
            nn.BatchNorm2d(16), # batch normalizationを行う層　引数は入力データのチャンネル数
            nn.LeakyReLU(0.2, inplace=True) # inplace=Trueにすることで、使用メモリを削減
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # decoderの層
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), # ２次元データの畳み込み演算を行う層　引数は入力のチャンネル数、出力のチャンネル数、カーネル(フィルタ)の大きさ
            nn.BatchNorm2d(256), # batch normalizationを行う層　引数は入力データのチャンネル数
            nn.Dropout2d(p=0.5), # 50%の割合でドロップアウトを実行
            nn.ReLU(inplace=True) # inplace=Trueにすることで、使用メモリを削減
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = torch.randn(64, 1, 512, 128) # (batch_size, num_channels, height, width)
        # encoder forward
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)
        h5 = self.conv5(h4)
        h6 = self.conv6(h5)
        # decoder forward
        dh1 = self.deconv1(h6)
        dh2 = self.deconv2(torch.cat((dh1, h5), dim=1))
        dh3 = self.deconv3(torch.cat((dh2, h4), dim=1))
        dh4 = self.deconv4(torch.cat((dh3, h3), dim=1))
        dh5 = self.deconv5(torch.cat((dh4, h2), dim=1))
        dh6 = self.deconv6(torch.cat((dh5, h1), dim=1))
        return dh6

# カーネルサイズ3×3のUnetのモデルを定義
class Unet_kernel3(nn.Module):
    def __init__(self):
        super(Unet_kernel3, self).__init__()
        # 畳み込み層のパラメータを指定
        kernel_size = 3
        stride = 2
        padding = 1
        # encoderの層
        self.conv_block1 = ConvBlock(1, 16, kernel_size, stride, padding)
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
        self.deconv_block6 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size, stride, padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = torch.randn(64, 1, 512, 128) # (batch_size, num_channels, height, width)
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
        dh6 = self.deconv_block6(torch.cat((dh5, h1), dim=1))
        return dh6

# カーネルサイズ5×5のUnetのモデルを定義
class Unet_kernel5(nn.Module):
    def __init__(self):
        super(Unet_kernel5, self).__init__()
        # 畳み込み層のパラメータを指定
        kernel_size = 5
        stride = 2
        padding = 2
        # encoderの層
        self.conv_block1 = ConvBlock(1, 16, kernel_size, stride, padding)
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
        self.deconv_block6 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size, stride, padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = torch.randn(64, 1, 512, 128) # (batch_size, num_channels, height, width)
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
        dh6 = self.deconv_block6(torch.cat((dh5, h1), dim=1))
        return dh6

# カーネルサイズ3×3のMulti-Channel Denoising Unetのモデルを定義
class MCDUnet_kernel3(nn.Module):
    def __init__(self):
        super(MCDUnet_kernel3, self).__init__()
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
        self.deconv_block6 = nn.Sequential(
            nn.ConvTranspose2d(32, 8, kernel_size, stride, padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = torch.randn(64, 1, 512, 128) # (batch_size, num_channels, height, width)
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
        dh6 = self.deconv_block6(torch.cat((dh5, h1), dim=1))
        return dh6


def main():
    # model = Unet_kernel3().cuda()
    model = MCDUnet_kernel3()
    tensor = torch.rand(8, 8, 257, 513) # (batch_size, channels, freq_bin, time_steps)
    output = model(tensor)
    print(output.shape)

    # summary(model, (1, 257, 513))


if __name__ == "__main__":
    main()
