import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3次元描画用
import seaborn as sns
sns.set()
from scipy.linalg import eigh



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
    # # 固有値分解を実施して最大固有値を与える固有ベクトルを取得
    # w, v = np.linalg.eigh(Rs)
    # """w: (freq_bins, num_microphones), v: (freq_bins, num_microphones, num_microphones)"""
    # steering_vector = v[:, :, -1]
    # """steering_vector: (freq_bins, num_microphones)"""

    # 固有値分解を実施して最大固有値を与える固有ベクトルを取得
    freq_bins, num_microphones, _ = Rs.shape
    steering_vector = np.empty((freq_bins, num_microphones), dtype=np.complex)
    # 各周波数ビンごとに処理
    for f in range(int(freq_bins)):         
        # 一般化固有値分解
        eigenvalues, eigenvectors = eigh(Rs[f, :, :])
        """eigenvalues: (num_microphones, ), eigenvectors: (num_microphones, num_microphones)"""
        steering_vector[f, :] = eigenvectors[:, -1] # 最大固有値に対応する固有ベクトルを取得
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
# def mvdr_beamformer_old(complex_spec, steering_vector, Rn):
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
    complex_spec: マイクロホン入力信号 (num_microphones, freq_bins, time_frames)
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

# MVDRビームフォーマ（共分散行列のみから計算、話者2人用）
def mvdr_beamformer_two_speakers(complex_spec, Rs, Ri, Rn):
    """
    complex_spec: マイクロホン入力信号 (num_microphones, freq_bins, time_steps)
    Rs: 目的話者の共分散行列 (freq_bins, num_microphones, num_microphones)
    Ri: 干渉話者の共分散行列（freq_bins, num_microphones, num_microphones） 
    Rn: 雑音の共分散行列 (freq_bins, num_microphones, num_microphones)
    """
    # 共分散行列の逆行列を計算する
    Ri_plus_Rn_inverse = np.linalg.pinv(Ri + Rn)
    """Ri_plus_Rn_inverse: (freq_bins, num_microphones, num_microphones)"""
    # 分離フィルタを計算する
    Ri_plus_Rn_inverse_Rs = np.einsum("fmi,fin->fmn", Ri_plus_Rn_inverse, Rs)
    """Ri_plus_Rn_inverse_Rs: (freq_bins, num_microphones, num_microphones)"""
    w_mvdr = Ri_plus_Rn_inverse_Rs / np.maximum(np.trace(Ri_plus_Rn_inverse_Rs, axis1=-2, axis2=-1), 1.e-18)[:, np.newaxis, np.newaxis]
    """w_mvdr: (freq_bins, num_microphones, num_microphones)"""
    # マイクロホン入力信号に分離フィルタを掛けて目的音成分を推定
    estimated_target_complex_spec = np.einsum("fmn,mft->nft", np.conjugate(w_mvdr), complex_spec)
    """estimated_target_complex_spec: (num_microphones, freq_bins, time_frames)"""
    return estimated_target_complex_spec

# MVDRビームフォーマ（共分散行列のみから計算、オンライン版）
def mvdr_beamformer_online(complex_spec, Rs, Rn):
    """
    complex_spec: マイクロホン入力信号 (num_microphones, freq_bins, time_frames)
    Rs: 目的音の共分散行列 (freq_bins, num_microphones, num_microphones)
    Rn: 雑音の共分散行列 (freq_bins, num_microphones, num_microphones)
    """
    # 逆行列を計算する代わりに逆行列の補題を用いて処理量を削減
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


# def getSteeringVector(angle, frequency, mic_alignments, freqs, sound_speed=340, is_use_far=False):
#         """
#         Generates a steering vector for given array geometry.
#         @param angle angle of sound source in rad
#         @param frequency frequency of the emitted sound in Hz
#         @param mode determines whether the returned vector will be testet against a covariance matrix
#                 or a cross-spectral matrix (possible values: 'cov', 'spec')
#         @return a numpy nx1 matrix, with n as amount of antennas
#         """
#         # TAMAGO-03のマイクロホンアレイのマイクロホン配置（単位はm）
#         mic_alignments = np.array(
#         [
#             [0.035, 0.0, 0.0],
#             [0.035/np.sqrt(2), 0.035/np.sqrt(2), 0.0],
#             [0.0, 0.035, 0.0],
#             [-0.035/np.sqrt(2), 0.035/np.sqrt(2), 0.0],
#             [-0.035, 0.0, 0.0],
#             [-0.035/np.sqrt(2), -0.035/np.sqrt(2), 0.0],
#             [0.0, -0.035, 0.0],
#             [0.035/np.sqrt(2), -0.035/np.sqrt(2), 0.0]
#         ])
#         mic_alignments = mic_alignments.T
#         """mic_alignments: (3D coordinates [m], num_microphones)"""

#         # 音源方向（音源が複数ある場合はリストに追加、目的音の音源方向は固定）
#         azimuth = [angle] # 方位角（1個目の音源, 2個目の音源）
#         elevation = [np.pi/6] # 仰角（1個目の音源, 2個目の音源）
#         # 音源の到来方向（HARK座標系に対応） [仰角θ, 方位角φ]
#         doas = np.array(
#         [[elevation[0], azimuth[0]], # １個目の音源 
# #         [elevation[1], azimuth[1]] # ２個目の音源
#         ])
#         # 音源の位置（単位球面上に存在すると仮定）
#         source_locations = np.zeros((3, doas.shape[0]), dtype=doas.dtype)
#         """source_locations: (xyz, num_sources)"""
#         source_locations[0,  :] = np.cos(doas[:, 1]) * np.cos(doas[:, 0]) # x = rcosφcosθ
#         source_locations[1,  :] = np.sin(doas[:, 1]) * np.cos(doas[:, 0]) # y = rsinφcosθ
#         source_locations[2,  :] = np.sin(doas[:, 0]) # z = rsinθ
#         # source_locations *= distance_mic_to_source
#         # source_locations += mic_array_loc[:, None] # マイクロホンアレイからの相対位置→絶対位置
#         # for i in range(doas.shape[0]):
#         #     x = source_locations[0, i]
#         #     y = source_locations[1, i]
#         #     z = source_locations[2, i]

#         # Far-field仮定（無限遠に音源が存在すると仮定）の場合
#         if is_use_far == True:
#             # 音源位置を正規化
#             norm_source_locations = source_locations / np.linalg.norm(source_locations, 2, axis=0, keepdims=True)
#             """norm_source_locations: (3D-coordinate(x,y,z)=3, num_sources)"""
#             # 位相を求める
#             steering_phase = np.einsum('k,ism,ism->ksm', 2.j*np.pi/sound_speed*freqs, norm_source_locations[...,None], mic_alignments[:, None, :])
#             """steering_phase: (freq_bins, num_sources, num_microphones)"""
#             # ステアリングベクトルを算出
#             steering_vector = 1./np.sqrt(n_channels)*np.exp(steering_phase)
#             """steering_vector: (freq_bins, num_sources, num_microphones)"""
#             return steering_vector
#         # Near-field仮定（音源がマイクロホン近くに存在すると仮定）の場合
#         else:
#             # 音源とマイクロホンの距離を求める
#             distance = np.sqrt(np.sum(np.square(source_locations[..., None]-mic_alignments[:, None, :]), axis=0))
#             """distance: (num_sources, num_microphones)"""
#             # 遅延時間 [sec]
#             delay = distance / sound_speed
#             """delay: (num_sources, num_microphones)"""
#             # ステアリングベクトルの位相を求める
#             steering_phase = np.einsum('k,sm->ksm', -2.j*np.pi*freqs, delay)
#             """steering_phase: (freq_bins, num_sources, num_microphones)"""
#             # 音量の減衰
#             steering_decay_ratio = 1./distance
#             # ステアリングベクトルを求める
#             steering_vector = steering_decay_ratio[None, ...]*np.exp(steering_phase)
#             # 大きさ1で正規化する
#             steering_vector = steering_vector / np.linalg.norm(steering_vector, 2, axis=2, keepdims=True)
#             """steering_vector: (freq_bins, num_sources, num_microphones)"""


#         # num_microphones = mic_alignments.shape[1]
#         # speedOfSound = 343.2 # m/s
#         # doa = np.matrix([[np.cos(angle)], [np.sin(angle)]], dtype=np.float128)
#         # steering_vector = np.matrix(np.empty((num_microphones, 1), dtype=np.complex256))
#         # for i in range(num_microphones):
#         #     # Using orthogonal projection, to get the propagation delay of the wavefront
#         #     # A simpler implementation, because doa is already a normalized vector
#         #     delay = (mic_alignments[:, i].T * doa) / speedOfSound # Apply dot product matrix style
#         #     steering_vector[i] = np.exp(-1j * 2 * np.pi * frequency * delay)
#         return steering_vector


# ステアリングベクトルを算出
def calculate_steering_vector(mic_alignments, angle, frequency, sound_speed=340, is_use_far=False):
    """
    mic_alignments: (3D-coordinate(x,y,z)=3, num_microphones(M))
    source_locations: (3D-coordinate(x,y,z)=3, num_sources(Ns))
    freqs: (freq_bins(Nk), )
    sound_speed: constant number
    is_use_far: Far -> True, Near -> False
    return: steering vector (Nk, Ns, M)
    """

    # 音源方向（音源が複数ある場合はリストに追加、目的音の音源方向は固定）
    azimuth = [angle] # 方位角（1個目の音源, 2個目の音源）
    elevation = [np.pi/6] # 仰角（1個目の音源, 2個目の音源）
    # 音源の到来方向（HARK座標系に対応） [仰角θ, 方位角φ]
    doas = np.array(
    [[elevation[0], azimuth[0]], # １個目の音源 
#         [elevation[1], azimuth[1]] # ２個目の音源
    ])
    # 音源の位置（単位球面上に存在すると仮定）
    source_locations = np.zeros((3, doas.shape[0]), dtype=doas.dtype)
    """source_locations: (xyz, num_sources)"""
    source_locations[0,  :] = np.cos(doas[:, 1]) * np.cos(doas[:, 0]) # x = rcosφcosθ
    source_locations[1,  :] = np.sin(doas[:, 1]) * np.cos(doas[:, 0]) # y = rsinφcosθ
    source_locations[2,  :] = np.sin(doas[:, 0]) # z = rsinθ

    # マイクロホン数を取得
    n_channels = np.shape(mic_alignments)[1]
    # 音源数を取得
    n_sources = np.shape(source_locations)[1]

    # Far-field仮定（無限遠に音源が存在すると仮定）の場合
    if is_use_far == True:
        # 音源位置を正規化
        norm_source_locations = source_locations / np.linalg.norm(source_locations, 2, axis=0, keepdims=True)
        """norm_source_locations: (3D-coordinate(x,y,z)=3, num_sources)"""
        # # 位相を求める
        # steering_phase = np.einsum('k,ism,ism->ksm', 2.j*np.pi/sound_speed*frequency, norm_source_locations[...,None], mic_alignments[:, None, :])
        # """steering_phase: (freq_bins, num_sources, num_microphones)"""
        # # ステアリングベクトルを算出
        # steering_vector = 1./np.sqrt(n_channels)*np.exp(steering_phase)
        # """steering_vector: (freq_bins, num_sources, num_microphones)"""
        # 位相を求める
        steering_phase = 2.j*np.pi/sound_speed*frequency * np.dot(norm_source_locations.T, mic_alignments)
        """steering_phase: (num_sources=1, num_microphones)"""
        # ステアリングベクトルを算出
        steering_vector = 1./np.sqrt(n_channels)*np.exp(steering_phase)
        """steering_vector: (num_sources=1, num_microphones)"""
    # Near-field仮定（音源がマイクロホン近くに存在すると仮定）の場合
    else:
        # 音源とマイクロホンの距離を求める
        distance = np.sqrt(np.sum(np.square(source_locations[:, :, None] - mic_alignments[:, None, :]), axis=0))
        """distance: (num_sources=1, num_microphones)"""
        # 遅延時間 [sec]
        delay = distance / sound_speed
        """delay: (num_sources=1, num_microphones)"""
        # # ステアリングベクトルの位相を求める
        # steering_phase = np.einsum('k,sm->ksm', -2.j*np.pi*frequency, delay)
        # """steering_phase: (freq_bins, num_sources, num_microphones)"""
        # ステアリングベクトルの位相を求める
        steering_phase = -2.j * np.pi * frequency * delay
        """steering_phase: (num_sources=1, num_microphones)"""
        # 音量の減衰
        steering_decay_ratio = 1./distance
        # ステアリングベクトルを求める
        steering_vector = steering_decay_ratio[None, ...] * np.exp(steering_phase)
        # 大きさ1で正規化する
        steering_vector = steering_vector / np.linalg.norm(steering_vector, 2, axis=2, keepdims=True)
        """steering_vector: (num_sources=1, num_microphones)"""
    
    # # 転置して縦ベクトルに
    # steering_vector = steering_vector.T
    # """steering_vector: (num_microphones, num_sources=1)"""
    return steering_vector

# Multiple signal classificationによる音源定位
def music(Rn, num_sources):
    """
    Rn: 雑音の共分散行列 (freq_bins, num_microphones, num_microphones)
    num_sources: 音源の数
    """

    # TAMAGO-03のマイクロホンアレイのマイクロホン配置（単位はm）
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
    mic_alignments = mic_alignments.T
    """mic_alignments: (3D coordinates [m], num_microphones)"""

    start_angle = -np.pi/2
    finish_angle = np.pi/2
    angle_step_num = 180 # 角度の解像度
    angle_step_size = (finish_angle - start_angle) / angle_step_num

    # # 固有値分解
    # eigenvalues, eigenvectors = eigh(Rs)
    # """eigenvalues: (freq_bins, num_microphones), eigenvectors: (freq_bins, num_microphones, num_microphones)"""
    # # 固有値の小さい固有ベクトル（雑音に対応する固有ベクトル）から順番に音源の数だけ選出
    # noise_eigenvectors = eigenvectors[:, :, :-num_sources]

    # 固有値分解を実施して最大固有値を与える固有ベクトルを取得
    freq_bins, num_microphones, _ = Rn.shape
    # steering_vector = np.empty((freq_bins, num_microphones), dtype=np.complex)
    # 空間スペクトル描画用
    all_music_spectrum = np.empty(freq_bins, dtype=object)
    # 各周波数ビンごとに処理
    for f in range(int(freq_bins)):         
        # 一般化固有値分解
        eigenvalues, eigenvectors = eigh(Rn[f, :, :])
        """eigenvalues: (num_microphones, ), eigenvectors: (num_microphones, num_microphones)"""
        # steering_vector[f, :] = eigenvectors[:, -1] # 最大固有値に対応する固有ベクトルを取得
        # 固有値を昇順から降順に並べ替え
        eigen_id = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[eigen_id]
        eigenvectors = eigenvectors[:, eigen_id]
        # 固有値の小さい固有ベクトルから順番に選択（雑音に対応する固有ベクトル）
        noise_eigenvectors = eigenvectors[:, num_sources+1:]
        """noise_eigenvectors: (num_microphones, num_microphones - num_sources)"""
        # # 固有値の小さい固有ベクトルから順番に選択（雑音に対応する固有ベクトル）
        # noise_eigenvectors = eigenvectors[:, :-num_sources]
        # """noise_eigenvectors: (num_microphones, num_microphones - num_sources)"""

        # 空間スペクトルの算出
        angle_steps = np.arange(start_angle, finish_angle, angle_step_size) # 空間スペクトルを算出する角度のリスト
        music_spectrum = np.empty(angle_step_num, dtype=np.float)
        for i in range(angle_step_num):
            virtual_steering_vector = calculate_steering_vector(mic_alignments, angle_steps[i], f, is_use_far=False)
            """virtual_steering_vector: (num_sources=1, num_microphones)"""
            virtual_steering_vector = np.squeeze(virtual_steering_vector)
            bunshi = np.einsum('m,m', np.conjugate(virtual_steering_vector), virtual_steering_vector)
            bunbo = np.einsum('m,mk,mk,m', np.conjugate(virtual_steering_vector), noise_eigenvectors, np.conjugate(noise_eigenvectors), virtual_steering_vector)
            music_spectrum[i] = 10 * np.log10(np.real(bunshi) / np.real(bunbo))
        # 角周波数ごとの空間スペクトルを格納
        if f == 0:
            all_music_spectrum = music_spectrum[np.newaxis, :]
        else:
            all_music_spectrum = np.append(all_music_spectrum, music_spectrum[np.newaxis, :], axis=0)

        # # 音源方向推定  
        # estimated_angle_index = np.argmax(music_spectrum)
        # estimated_angle_rad = angle_steps[estimated_angle_index]
        # estimated_angle_deg = estimated_angle_rad * 180 / np.pi
        # print(estimated_angle_deg) # 定位角

        # # 周波数ごとの空間スペクトルを可視化
        # # x_axis = np.linspace(-90, 90, angle_step_num)
        # plt.plot(angle_steps * 180 / np.pi, music_spectrum)
        # plt.xlabel("Angle[°]")
        # plt.ylabel("Spatial spectrum[dB]")
        # plt.savefig("test/localization_result/{}Hz.png".format(f*8000/256))
        # plt.clf()
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("Angle[°]")
    ax.set_ylabel("Frequency[Hz]")
    ax.set_zlabel("Spatial spectrum[dB]")
    # print((angle_steps * 180 / np.pi).shape)
    # print(np.linspace(0, 8000, freq_bins).shape)
    # print(all_music_spectrum.shape)

    # 8000Hzまで表示
    # ax.plot_wireframe(X, Y, all_music_spectrum)
    # ax.plot_wireframe(angle_steps * 180 / np.pi, np.linspace(0, 8000, freq_bins), all_music_spectrum)
    # X, Y = np.meshgrid(angle_steps * 180 / np.pi, np.linspace(0, 8000, freq_bins))
    # # 空間スペクトルの曲面をプロット rstrideとcstrideはステップサイズ，cmapは彩色，linewidthは曲面のメッシュの線の太さ，をそれぞれ表す
    # ax.plot_surface(X, Y, all_music_spectrum, rstride=1, cstride=1, cmap='hsv', linewidth=0.3) 
    # plt.savefig("test/spatial_spectrum/spatial_spectrum.png")
    # plt.clf()

    # 2000Hzまで表示
    X, Y = np.meshgrid(angle_steps * 180 / np.pi, (np.linspace(0, 8000, freq_bins))[:49])
    # 空間スペクトルの曲面をプロット rstrideとcstrideはステップサイズ，cmapは彩色，linewidthは曲面のメッシュの線の太さ，をそれぞれ表す
    ax.plot_surface(X, Y, all_music_spectrum[:49, :], rstride=1, cstride=1, cmap='hsv', linewidth=0.3) 
    plt.savefig("test/spatial_spectrum/spatial_spectrum.png")
    plt.clf()

    # 音源方向推定 
    estimated_angle_deg_list = []
    estimated_angle_index = np.argmax(all_music_spectrum, axis=1)[:49] # 2000Hz分
    # angle_steps_repeat = np.repeat(angle_steps[np.newaxis, :], 49, axis=0) * 180 / np.pi
    for index in estimated_angle_index:
        estimated_angle_rad = angle_steps[index]
        estimated_angle_deg = estimated_angle_rad * 180 / np.pi
        estimated_angle_deg_list.append(estimated_angle_deg)
    estimated_angle = np.average(estimated_angle_deg_list)
    print(estimated_angle)


        

if __name__ == "__main__":
    # num_microphones, freq_bins, time_frames
    num_microphones = 8
    sample_rate = 16000
    audio_length = 3
    freq_bins = 257
    time_frames = 301

    # mixed_complex_spec = np.random.rand(num_microphones, freq_bins, time_frames)
    # mask = np.random.rand(freq_bins, time_frames)
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