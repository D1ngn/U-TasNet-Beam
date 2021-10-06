import numpy as np
import soundfile as sf

from separation import bss_eval_sources, bss_eval_images


#　音声データをロードし、指定された秒数とサンプリングレートでリサンプル
def load_audio_file(file_path, length, sampling_rate=16000):
    data, sr = sf.read(file_path)
    # データが設定値よりも大きい場合は大きさを超えた分をカットする
    # データが設定値よりも小さい場合はデータの後ろを0でパディングする
    # シングルチャンネル(モノラル)の場合 (data.shape: [num_samples,])
    if data.ndim == 1:
        if len(data) > sampling_rate*length:
            data = data[:sampling_rate*length]
        else:
            data = np.pad(data, (0, max(0, sampling_rate*length - len(data))), "constant")
    # マルチチャンネルの場合 (data.shape: [num_samples, num_channels])
    elif data.ndim == 2:
        if data.shape[0] > sampling_rate*length:
            data = data[:sampling_rate*length, :]
        else:
            data = np.pad(data, [(0, max(0, sampling_rate*length-data.shape[0])), (0, 0)], "constant")
    else:
        print("number of audio channels are incorrect")
    return data

# 混合音声とモデルが推定した音声の質を評価(SDR, SIR, SARを算出)
def audio_eval(audio_length, target_audio_path, interference_audio_path, mixed_audio_path, estimated_audio_path):
    """
    bss_eval_sourcesとbss_eval_imagesに関しては
    「http://bass-db.gforge.inria.fr/bss_eval/」
    を参照
    """

    target = load_audio_file(target_audio_path, audio_length)[np.newaxis, :]
    interference = load_audio_file(interference_audio_path, audio_length)[np.newaxis, :]
    mixed = load_audio_file(mixed_audio_path, audio_length)[np.newaxis, :]
    estimated = load_audio_file(estimated_audio_path, audio_length)[np.newaxis, :]

    reference = np.concatenate([target, interference], 0) # 目的音と外的雑音を結合する
    mixed = np.concatenate([mixed, mixed], 0) # referenceと同じ形になるように結合
    estimated = np.concatenate([estimated, estimated], 0) # referenceと同じ形になるように結合

    # シングルチャンネル用 (シングルチャンネルの場合音声はshape:[1, num_samples]の形式)
    if target.ndim == 2:
        mixed_result = bss_eval_sources(reference, mixed) # 混合音声のSDR, SIR, SARを算出
        reference_result = bss_eval_sources(reference, estimated) # モデルが推定した音声のSDR, SIR, SARを算出
        print("SDR_mix: {:.3f}, SIR_mix: {:.3f}, SAR_mix: {:.3f}".format(mixed_result[0][0], mixed_result[1][0], mixed_result[2][0]))
        print("SDR_pred: {:.3f}, SIR_pred: {:.3f}, SAR_pred: {:.3f}".format(reference_result[0][0], reference_result[1][0], reference_result[2][0]))

    # マルチチャンネル用 (マルチチャンネルの場合音声はshape:[1, num_samples, num_channels]の形式)
    elif target.ndim == 3:
        mixed_result = bss_eval_images(reference, mixed) # 混合音声のSDR, SIR, SARを算出
        reference_result = bss_eval_images(reference, estimated) # モデルが推定した音声のSDR, SIR, SARを算出
        print("SDR_mix: {:.3f}, SIR_mix: {:.3f}, SAR_mix: {:.3f}".format(mixed_result[0][0], mixed_result[2][0], mixed_result[3][0]))
        print("SDR_pred: {:.3f}, SIR_pred: {:.3f}, SAR_pred: {:.3f}".format(reference_result[0][0], reference_result[2][0], reference_result[3][0]))

    else:
        print("number of audio channels are incorrect")


if __name__ == "__main__":

    audio_length = 3

    # 目的音
    target_audio_path = "../test/target_voice.wav"
    # target_audio_path = "../../AudioDatasets/NoisySpeechDetabase/clean_testset_wav_16kHz/p232_013.wav"
    # target_audio_path = "./sample_audio_multi/shokudo_rec1_split_0.wav" # マルチチャンネルテスト用
    # 外的雑音
    interference_audio_path = "../test/interference_audio.wav"
    # interference_audio_path = "../../AudioDatasets/NoisySpeechDetabase/interference_testset_wav_16kHz/p232_013.wav"
    # interference_audio_path = "./sample_audio_multi/shokudo_rec1_split_1.wav" # マルチチャンネルテスト用
    # 混合音
    mixed_audio_path = "../test/mixed_audio.wav"
    # mixed_audio_path = "../../AudioDatasets/NoisySpeechDetabase/noisy_testset_wav_16kHz/p232_013.wav"
    # mixed_audio_path = "./sample_audio_multi/shokudo_rec1_split_3.wav" # マルチチャンネルテスト用
    # モデルが推定した音声
    estimated_audio_path = "../test/estimated_voice.wav"
    # estimated_audio_path = "../../speech_denoising_DCUnet/test0822/test_RefineSpectrogramUnet_0822.wav"
    # estimated_audio_path = "./sample_audio_multi/shokudo_rec1_split_2.wav" # マルチチャンネルテスト用

    audio_eval(audio_length, target_audio_path, interference_audio_path, mixed_audio_path, estimated_audio_path)
