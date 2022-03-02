import torch


# 参考：「https://github.com/seorim0/DCCRN-with-various-loss-functions/blob/main/tools_for_loss.py」
# 2つのベクトルの内積を計算
def calc_inner_product(s1, s2):
    # norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True)) # L2 norm
    # norm = torch.norm(s1*s2, 1, keepdim=True)
    inner_product = torch.sum(s1 * s2, dim=-1, keepdim=True)
    return inner_product

def snr_loss(clean, estimated, eps=1e-8):
    clean_clean_inner_product = calc_inner_product(clean, clean)
    noise_noise_inner_product = calc_inner_product(estimated - clean, estimated - clean)
    snr = 10 * torch.log10(clean_clean_inner_product**2 / (noise_noise_inner_product**2 + eps) + eps) # epsはー∞、+∞に発散するのを防ぐため
    # チャンネル間の平均をとって返す（SNRを最大化したいため、-SNRを最小化するようにマイナスをつける）
    loss = -torch.mean(snr)
    return loss

def si_snr_loss(clean, estimated, eps=1e-8):
    """
    clean: (num_channels, num_samples)
    estimated: (num_channels, num_samples)
    """
    # clean = remove_dc(clean)
    # estimated = remove_dc(estimated)
    estimated_clean_inner_product = calc_inner_product(estimated, clean)
    clean_clean_inner_product = calc_inner_product(clean, clean)
    s_target = (estimated_clean_inner_product / (clean_clean_inner_product + eps)) * clean # epsは0除算を避けるため
    e_noise = estimated - s_target
    target_inner_product = calc_inner_product(s_target, s_target)
    noise_inner_product = calc_inner_product(e_noise, e_noise)
    snr = 10 * torch.log10((target_inner_product) / (noise_inner_product + eps) + eps) # epsはー∞、+∞に発散するのを防ぐため
    """snr: (num_channles, value)"""
    # チャンネル間の平均をとって返す（SI-SNRを最大化したいため、-SI-SNRを最小化するようにマイナスをつける）
    loss = -torch.mean(snr)
    return loss

# snr_lossと変わらない
def sdr_loss(s1, s2, eps=1e-8):
    sn = calc_inner_product(s1, s1)
    sn_m_shn = calc_inner_product(s1 - s2, s1 - s2)
    loss = 10 * torch.log10(sn**2 / (sn_m_shn**2 + eps))
    return torch.mean(loss)

# si_snr_lossと変わらない
def si_sdr_loss(clean, estimated, eps=1e-8):
    # SI-SDRの元論文：「https://www.merl.com/publications/docs/TR2019-013.pdf」
    """
        Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
        Args:
            clean: numpy.ndarray, [..., T]
            estimated: numpy.ndarray, [..., T]
        Returns:
            SI-SDR
        [1] SDR– Half- Baked or Well Done?
        http://www.merl.com/publications/docs/TR2019-013.pdf
        >>> np.random.seed(0)
        >>> clean = np.random.randn(100)
        >>> si_sdr(clean, clean)
        inf
        >>> si_sdr(clean, clean * 2)
        inf
        >>> si_sdr(clean, np.flip(clean))
        -25.127672346460717
        >>> si_sdr(clean, clean + np.flip(clean))
        0.481070445785553
        >>> si_sdr(clean, clean + 0.5)
        6.3704606032577304
        >>> si_sdr(clean, clean * 2 + 1)
        6.3704606032577304
        >>> si_sdr([1., 0], [0., 0])  # never predict only zeros
        nan
        >>> si_sdr([clean, clean], [clean * 2 + 1, clean * 1 + 0.5])
        array([6.3704606, 6.3704606])
        :param clean:
        :param estimated:
        :param eps:
        """

    clean_energy = torch.sum(clean ** 2, axis=-1, keepdims=True)

    # This is $\alpha$ after Equation (3) in [1].
    optimal_scaling = torch.sum(clean * estimated, axis=-1, keepdims=True) / clean_energy + eps

    # This is $e_{\text{target}}$ in Equation (4) in [1].
    projection = optimal_scaling * clean

    # This is $e_{\text{res}}$ in Equation (4) in [1].
    noise = estimated - projection

    ratio = torch.sum(projection ** 2, axis=-1) / torch.sum(noise ** 2, axis=-1) + eps
    
    # チャンネル間の平均をとって返す（SI-SDRを最大化したいため、-SI-SDRを最小化するようにマイナスをつける）
    ratio = torch.mean(ratio)
    loss = -10 * torch.log10(ratio + eps)
    return loss

# model outputs speech and noise
def weighted_sdr_loss(mixed, speech, noise, est_speech, est_noise, eps=1e-8):
    """
    mixed: (num_channels, num_samples)
    speech: (num_channels, num_samples)
    noise: (num_channels, num_samples)
    est_speech: (num_channels, num_samples)
    est_noise: (num_channels, num_samples)
    """
    # to time-domain waveform
#     y_true_ = torch.squeeze(y_true_, 1)
#     mixed = torch.squeeze(x_, 1)

    mixed = mixed.flatten(1)
    speech = speech.flatten(1)
    noise = noise.flatten(1)
    est_speech = est_speech.flatten(1)
    est_noise = est_noise.flatten(1)

    def sdr_fn(true, pred, eps=1e-8):
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return -(num / (den + eps))

    # weighted SDRの算出
    a = torch.sum(speech**2, dim=1) / (torch.sum(speech**2, dim=1) + torch.sum(noise**2, dim=1) + eps)
    wSDR = a * sdr_fn(speech, est_speech) + (1 - a) * sdr_fn(noise, est_noise)
    
    return torch.mean(wSDR)

#############################Multi-channel ConvTasnet用###############################
# 相関係数を算出
def calc_corr_coef(x, y, eps=1e-8):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    coef = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + eps)
    return coef

# チャンネルごとに順序がバラバラの発話を揃える（話者が2人のときのみに対応）
def reorder_speech(speech, speech_idx):
    """speech: (batch_size, num_speakers, num_channels, num_samples)"""
    another_speech_idx = speech_idx - 1
    # 空のテンソルを用意
    reorderd_speech = torch.empty_like(speech)
    """reorderd_speech: (batch_size, num_speakers, num_channels, num_samples)"""
    # 一人目の話者の発話を格納
    for ch, idx in enumerate(speech_idx):
        reorderd_speech[:, 0, ch, :] = speech[:, idx, ch, :]
    # 二人目の話者の発話を格納
    for ch, idx in enumerate(another_speech_idx):
        reorderd_speech[:, 1, ch, :] = speech[:, idx, ch, :]
    return reorderd_speech

# inter-channel permutation problemの解決
def solve_inter_channel_permutation_problem(speech, ref_ch=0):
    """speech: (batch_size, num_speakers, num_channels, num_samples)"""
    ref_ch_speech = speech[:, 0, ref_ch, :] # 1番目の発話の基準チャンネルの発話を取得
    """ref_ch_speech: (batch_size, num_samples)"""
    ref_speech_idx_list = []
    for ch in range(speech.shape[2]):
        coef_list = []
        for speech_idx in range(speech.shape[1]):      
            coef = calc_corr_coef(ref_ch_speech, speech[:, speech_idx, ch, :])
            coef_list.append(coef)
        ref_speech_idx_list.append(torch.argmax(torch.tensor(coef_list)))
    # 相関係数をもとに取得したインデックスにしたがって発話を並び替え
    reordered_speech = reorder_speech(speech, torch.tensor(ref_speech_idx_list))
    """reordered_speech: (batch_size, num_speakers, num_channels, num_samples)"""
    return reordered_speech

# Permutation Invariant Training（PIT）に基づくlossを算出
def pit_loss(target_speech, estimated_speech):
    # 2人の発話の順序を並び替えながら正解データとのSNR lossを算出し、小さい方のSNR lossを使用（学習時のみ）
    # 愚直に並び替え
    another_estimated_speech = torch.empty_like(estimated_speech)
    another_estimated_speech[:, 0, :, :] = estimated_speech[:, 1, :, :]
    another_estimated_speech[:, 1, :, :] = estimated_speech[:, 0, :, :]
    # SNR lossを算出して比較
    snr_loss_1 = snr_loss(target_speech, estimated_speech)
    snr_loss_2 = snr_loss(target_speech, another_estimated_speech)
    if snr_loss_1 <= snr_loss_2:
        loss = snr_loss_1
    else:
        loss = snr_loss_2
    return loss

#############################提案手法用###############################
# 信号の複素スペクトログラムのみから共分散行列（空間相関行列）を推定（バッチ次元追加版）
def estimate_covariance_matrix_sig_batch_torch(complex_spec):
    """
    complex_spec: 入力複素スペクトログラム (batch_size, num_channles, freq_bins, time_frames)
    """
    # 空間相関行列を算出    
    spatial_covariance_matrix = torch.einsum("bmft,bnft->bfmn", complex_spec, torch.conj(complex_spec))
    """spatial_covariance_matrix: (batch_size, freq_bins, num_microphones, num_microphones)"""
    # 固有値分解をして半正定値行列に変換（eighはPyTorchのバージョン1.9以降で実装）
    eigenvalues, eigenvectors = torch.linalg.eigh(spatial_covariance_matrix) #   numpy.linalg.eighとtorch.linalg.eighの挙動の違いを確認する必要がある TODO
    """eigenvalues: (batch_size, freq_bins, num_microphones), eigenvectors: (batch_size, freq_bins, num_microphones, num_microphones)"""
    # 固有値が0より小さい場合は微小な数に置き換える
    mask = eigenvalues.ge(0) # 0以上の部分をTrue、0より小さい部分をFlaseとしたマスクを生成
    eigenvalues = eigenvalues * mask
    eigenvalues = eigenvalues.masked_fill(mask==False, 1e-18) # 固有値が0より小さい部分（False）を微小な数に置き換える
#     eigenvalues[eigenvalues < 1e-18] = 1e-18 # 固有値が0より小さい場合は微小な数に置き換える # in-place operationはbackward時にエラーが出る（これは使えない）
    spatial_covariance_matrix = torch.einsum("bfmi,bfi,bfni->bfmn", eigenvectors, eigenvalues, torch.conj(eigenvectors))
    """spatial_covariance_matrix: (batch_size, freq_bins, num_microphones, num_microphones)"""
    return spatial_covariance_matrix

# 行列の対角成分の和を算出する関数（torch.trace()では3次元以上のテンソルを入力できないため）
def trace(input, axis1=0, axis2=1):
    # 参考：「https://github.com/pytorch/pytorch/issues/52668」
    """
    >>> torch.__version__
    '1.9.0.dev20210222+cpu'
    >>> x = torch.arange(1., 10.).view(3, 3)
    >>> x
    tensor([[1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]])
    >>> torch.trace(x)
    tensor(15.)
    >>> torch.trace(x.view(1, 3, 3))
    Traceback (most recent call last):
    ...
    RuntimeError: trace: expected a matrix, but got tensor with dim 3
    >>> trace(x)
    tensor(15.)
    >>> trace(x.view(3, 3, 1), axis1=0, axis2=1)
    tensor([15.])
    >>> trace(x.view(1, 3, 3), axis1=2, axis2=1)
    tensor([15.])
    >>> trace(x.view(3, 1, 3), axis1=0, axis2=2)
    tensor([15.])
    """
#     assert input.shape[axis1] == input.shape[axis2], input.shape

    shape = list(input.shape)
    strides = list(input.stride())
    strides[axis1] += strides[axis2]

    shape[axis2] = 1
    strides[axis2] = 0

    input = torch.as_strided(input, size=shape, stride=strides)
    return input.sum(dim=(axis1, axis2))

# スパースかつ良条件（条件数が少ない問題）の共分散行列を推定
def condition_covariance(x, device, eps=1e-6):
    """see https://stt.msu.edu/users/mauryaas/Ashwini_JPEN.pdf (2.3)"""
    """x: (batch_size, freq_bins, num_microphones, num_microphones)"""
    scale = eps * trace(x) /  x.shape[-1]
    scaled_eye = torch.eye(x.shape[-1]).to(device) * scale
    return (x + scaled_eye) / (1 + eps)

# 推定した空間相関行列と正解の空間相関行列のL1 lossを算出するオリジナルの損失関数
def scm_loss(clean, estimated, device, flag, num_channels=8):
    """
    clean: (batch_size*num_channels, num_samples)
    estimated: (batch_size*num_channels, num_samples)
    device: 'cpu' or 'cuda:{}'
    flag: 'speech' or 'noise'
    num_channels: Number of microphone channels (default: 8)
    """
    # 音声波形を複素スペクトログラムに変換
    clean_complex_spec = torch.stft(input=clean, n_fft=512, hop_length=160, normalized=False, return_complex=True)
    """clean_complex_spec: (batch_size*num_channels, freq_bins, time_frames)"""
    estimated_complex_spec = torch.stft(input=estimated, n_fft=512, hop_length=160, normalized=False, return_complex=True)
    """estimated_complex_spec: (batch_size*num_channels, freq_bins, time_frames)"""
    # バッチサイズの次元とチャンネル数の次元を元に戻す
    clean_complex_spec = clean_complex_spec.contiguous().view(-1, num_channels, clean_complex_spec.shape[1], clean_complex_spec.shape[2])
    """clean_complex_spec: (batch_size, num_channels, freq_bins, time_frames)"""
    estimated_complex_spec = estimated_complex_spec.contiguous().view(-1, num_channels, estimated_complex_spec.shape[1], estimated_complex_spec.shape[2])
    """estimated_complex_spec: (batch_size, num_channels, freq_bins, time_frames)""" 
    # 空間相関行列を算出
    clean_spatial_covariance_matrix = estimate_covariance_matrix_sig_batch_torch(clean_complex_spec)
    """clean_spatial_covariance_matrix: (batch_size, freq_bins, num_microphones, num_microphones)"""
    estimated_spatial_covariance_matrix = estimate_covariance_matrix_sig_batch_torch(estimated_complex_spec)
    """estimated_spatial_covariance_matrix: (batch_size, freq_bins, num_microphones, num_microphones)"""
    # ロスを算出
    # 雑音の空間相関行列を算出する場合以下の処理がないと性能が大きく落ちる
    if flag == 'noise':
        clean_spatial_covariance_matrix = condition_covariance(clean_spatial_covariance_matrix, device, 1e-6)
        estimated_spatial_covariance_matrix = condition_covariance(estimated_spatial_covariance_matrix, device, 1e-6)
    # MAE loss
    loss = torch.mean(torch.abs(estimated_spatial_covariance_matrix - clean_spatial_covariance_matrix)) / num_channels 
    return loss

if __name__ == "__main__":
    B, P, C, T = 8, 2, 8, 48000
    # target_speech = torch.randint(4, (B, C, T))
    # estimated_speech = torch.randint(4, (B, C, T))
    target_speech = torch.rand((B, P, C, T))
    """target_speech: (batch_size, num_speakers, num_channels, num_samples)"""
    estimated_speech = torch.rand((B, P, C, T))
    """estimated_speech: (batch_size, num_speakers, num_channels, num_samples)"""
    
    reordered_estimated_speech = solve_inter_channel_permutation_problem(target_speech, estimated_speech)
    loss = pit_loss(target_speech, reordered_estimated_speech)
    print(loss)
    
