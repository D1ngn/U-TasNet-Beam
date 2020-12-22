import numpy as np
from scipy.linalg import eigh

from utils import load_audio_file, wave_to_spec_multi


# マスクと入力信号から共分散行列（空間相関行列）を推定
def estimate_covariance_matrix(complex_spec, mask):
    """
    complex_spec: マイクロホン入力信号 (num_microphones, freq_bins, time_frames)
    mask: 音源方向情報を含む時間周波数マスク (freq_bins, time_frames)
    """
    # 空間相関行列を算出
    spatial_covariance_matrix = np.einsum("ft,mft,nft->fmn", mask, complex_spec, np.conjugate(complex_spec))
    """spatial_covariance_matrix: (freq_bins, num_microphones, num_microphones)"""
    # 正規化
    sum_mask = np.maximum(np.sum(mask, axis=-1), 1e-18)[:, np.newaxis, np.newaxis]
    """sum_mask: (freq_bins, num_microphones=1, num_microphones=1)"""
    spatial_covariance_matrix /= sum_mask
    """spatial_covariance_matrix: (freq_bins, num_microphones, num_microphones)"""
    # 固有値分解をして半正定値行列に変換
    eigenvalues, eigenvectors = np.linalg.eigh(spatial_covariance_matrix)
    """eigenvalues: (freq_bins, num_microphones), eigenvectors: (freq_bins, num_microphones, num_microphones)"""
    eigenvalues[np.real(eigenvalues) < 1e-18] = 1e-18 # 固有値が0より小さい場合は0に置き換える
    spatial_covariance_matrix = np.einsum("fmi,fi,fni->fmn", eigenvectors, eigenvalues, np.conjugate(eigenvectors))
    """spatial_covariance_matrix: (freq_bins, num_microphones, num_microphones)"""
    return spatial_covariance_matrix

# スパースかつ良条件（条件数が少ない問題）の共分散行列を推定
def condition_covariance(x, gamma):
    """see https://stt.msu.edu/users/mauryaas/Ashwini_JPEN.pdf (2.3)"""
    scale = gamma * np.trace(x) / x.shape[-1]
    scaled_eye = np.eye(x.shape[-1]) * scale
    return (x + scaled_eye) / (1 + gamma)

# 音源のスパース性を仮定し、共分散行列からステアリングベクトルを推定する
def estimate_steering_vector(Rs):
    """
    Rs: 共分散行列 (freq_bins, num_microphones, num_microphones)
    """
    # 固有値分解を実施して最大固有値を与える固有ベクトルを取得
    w, v = np.linalg.eigh(Rs)
    """w: (freq_bins, num_microphones), v: (freq_bins, num_microphones, num_microphones)"""
    steering_vector = v[:, :, -1]
    """steering_vector: (freq_bins, num_microphones)"""
    return steering_vector

# スパース性に基づく分離（テスト用）
def sparse(complex_spec, mask):
    """
    complex_spec: (num_microphones, freq_bins, time_frames)
    mask: (freq_bins, time_frames)
    """
    estimated_target_complex_spec = np.einsum("kt,mkt->mkt", mask, complex_spec)
    """estimated_target_complex_spec: (num_microphones, freq_bins, time_frames)"""
    return estimated_target_complex_spec

# 遅延和アレイビームフォーマ
def ds_beamformer(complex_spec, steering_vector):
    """
    complex_spec: (num_microphones, freq_bins, time_frames)
    steering_vector: (freq_bins, num_microphones)
    """
    s_hat = np.einsum("fm,mft->ft", np.conjugate(steering_vector), complex_spec)
    """s_hat: (num_sources, freq_bins, time_frames)"""
    # ステアリングベクトルを掛ける
    c_hat = np.einsum("ft,fm->mft", s_hat, steering_vector)
    """c_hat: (num_microphones, freq_bins, time_frames)"""
    return c_hat

# # MVDRビームフォーマ（ステアリングベクトルを用いて計算）
# def mvdr_beamformer(complex_spec, steering_vector, Rn):
#     """
#     complex_spec: マイクロホン入力信号 (num_microphones, freq_bins, time_frames)
#     steering_vector: 目的音のステアリングベクトル (freq_bins, num_microphones)
#     Rn: 雑音の共分散行列 (freq_bins, num_microphones, num_microphones)
#     """
#     # 共分散行列の逆行列を計算する
#     Rn_inverse = np.linalg.pinv(Rn)
#     """Rn_inverse: (freq_bins, num_microphones, num_microphones)"""
#     # 分離フィルタを計算する
#     Rn_inverse_a = np.einsum("fmn,fn->fm", Rn_inverse, steering_vector) # 分子
#     """Rn_inverse_a: (freq_bins, num_microphones)"""
#     a_H_Rn_inverse_a = np.einsum("fn,fn->f", np.conjugate(steering_vector), Rn_inverse_a) # 分母
#     """a_H_Rn_inverse_a: (freq_bins, )"""
#     w_mvdr = Rn_inverse_a / np.maximum(a_H_Rn_inverse_a, 1.e-18)[:, np.newaxis]
#     """w_mvdr: (freq_bins, num_microphones)"""
#     # 分離フィルタを掛ける
#     s_hat = np.einsum("fm,mft->ft", np.conjugate(w_mvdr), complex_spec)
#     """s_hat: (freq_bins, time_frames)"""
#     # ステアリングベクトルを掛ける（マイクロホン入力信号中の目的音成分を推定）
#     c_hat = np.einsum("ft,fm->mft", s_hat, steering_vector)
#     """c_hat: (num_microphones, freq_bins, time_frames)"""
#     return c_hat

# MVDRビームフォーマ（共分散行列のみから計算）
def mvdr_beamformer(complex_spec, Rs, Rn):
    """
    complex_spec: マイクロホン入力信号 (num_microphones, freq_bins, time_steps)
    Rs: 目的音の共分散行列 (freq_bins, num_microphones, num_microphones)
    Rn: 雑音の共分散行列 (freq_bins, num_microphones, num_microphones)
    """
    # 共分散行列の逆行列を計算する
    Rn_inverse = np.linalg.pinv(Rn)
    """Rn_inverse: (freq_bins, num_microphones, num_microphones)"""
    # 分離フィルタを計算する
    Rn_inverse_Rs = np.einsum("fmi,fin->fmn", Rn_inverse, Rs)
    """Rn_inverse_Rs: (freq_bins, num_microphones, num_microphones)"""
    w_mvdr = Rn_inverse_Rs / np.maximum(np.trace(Rn_inverse_Rs, axis1=-2, axis2=-1), 1.e-18)[:, np.newaxis, np.newaxis]
    """w_mvdr: (freq_bins, num_microphones, num_microphones)"""
    # マイクロホン入力信号に分離フィルタを掛けて目的音成分を推定
    estimated_target_complex_spec = np.einsum("fmn,mft->nft", np.conjugate(w_mvdr), complex_spec)
    """estimated_target_complex_spec: (num_microphones, freq_bins, time_frames)"""
    return estimated_target_complex_spec

# GEV（MaxSNR）ビームフォーマ
def gev_beamformer(complex_spec, Rs, Rn):
    """
    complex_spec: マイクロホン入力信号 (num_microphones, freq_bins, time_frames)
    Rs: 目的音の共分散行列 (freq_bins, num_microphones, num_microphones)
    Rn: 雑音の共分散行列 (freq_bins, num_microphones, num_microphones)
    """
    freq_bins, num_microphones, _ = Rs.shape
    gev_filter = np.empty((freq_bins, num_microphones), dtype=np.complex)
    # 各周波数ビンごとに処理
    for f in range(int(freq_bins)):         
        # 一般化固有値分解
        eigenvalues, eigenvectors = eigh(Rs[f, :, :], Rn[f, :, :])
        """eigenvalues: (num_microphones, ), eigenvectors: (num_microphones, num_microphones)"""
        gev_filter[f, :] = eigenvectors[:, -1] # 最大固有値に対応する固有ベクトルを取得
    """gev_filter: (freq_bins, num_microphones)"""
    # フィルタの大きさの不定性を解決するための処理
    # 分離フィルタをマイクロホン入力信号中の目的音のみにかけたときの出力信号が
    # マイクロホン入力信号中の真の目的音に近づくように不定性の係数を求める
    # 参考：「https://www.tara.tsukuba.ac.jp/~maki/reprint/Araki/sa07icasspI-41-I-44.pdf」
    Rs_w = np.einsum("fmn,fn->fm", Rs, gev_filter)
    """Rs_w: (freq_bins, num_microphones)"""
    indef_coeff = Rs_w / np.einsum("fm,fm->f", np.conjugate(gev_filter), Rs_w)[:, np.newaxis]
    """indef_coeff: (freq_bins, num_microphones)"""
    w_gev = indef_coeff[:, np.newaxis, :] * gev_filter[:, :, np.newaxis]
    """w_gev: (freq_bins, num_microphones, num_microphones)"""
    # マイクロホン入力信号に分離フィルタを掛けて目的音成分を推定
    estimated_target_complex_spec = np.einsum("fmn,mft->nft", np.conjugate(w_gev), complex_spec)
    """estimated_target_complex_spec: (num_microphones, freq_bins, time_frames)"""

    return estimated_target_complex_spec

# マルチチャンネルフィナーフィルタを実行
def mwf(complex_spec, Rs, Rn):
    """
    complex_spec: (num_microphones, freq_bins, time_frames)
    Rs: (freq_bins, num_microphones, num_microphones)
    Rn: (freq_bins, num_microphones, num_microphones)
    """
    # 入力信号に対する共分散行列の逆行列を計算
    Rx_inverse = np.linalg.pinv(Rs + Rn)
    """Rx_inverse: (freq_bins, num_microphones, num_microphones)"""
    # フィルタ生成
    W_mwf = np.einsum("fmi,fin->fmn", Rx_inverse, Rs)
    """W_mwf: (freq_bins, num_microphones, num_microphones)"""
    # フィルタを掛ける
    estimated_target_complex_spec = np.einsum("fim,ift->mft", np.conjugate(W_mwf), complex_spec)
    """c_hat: (num_microphones, freq_bins, time_frames)"""
    return estimated_target_complex_spec


if __name__ == "__main__":
    # num_microphones, freq_bins, time_steps
    num_microphones = 8
    sample_rate = 16000
    audio_length = 3
    freq_bins = 257
    time_steps = 301

    # mixed_complex_spec = np.random.rand(num_microphones, freq_bins, time_steps)
    # mask = np.random.rand(freq_bins, time_steps)
    # target_covariance_matrix = estimate_covariance_matrix(mixed_complex_spec, mask)
    # noise_covariance_matrix = estimate_covariance_matrix(mixed_complex_spec, mask)
    # # print(target_covariance_matrix.shape)
    # estimated_target_complex_spec = gev_beamformer(mixed_complex_spec, target_covariance_matrix, noise_covariance_matrix)
    # print(estimated_target_complex_spec.shape)

    file_path = "./test/p257_006/p257_006_mixed_azimuth60.wav"
    mixed_audio_data = load_audio_file(file_path, audio_length, sample_rate)
    # print(mixed_audio_data.min())
    # print(mixed_audio_data.max())
    mixed_audio_data = mixed_audio_data.transpose(1, 0)
    mixed_complex_spec = wave_to_spec_multi(mixed_audio_data, sample_rate, fft_size=512, hop_length=160)
    print(mixed_complex_spec.min())
    print(mixed_complex_spec.max())