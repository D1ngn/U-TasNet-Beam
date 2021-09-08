import os
import numpy as np
import soundfile as sf


# 混合音声とモデルが推定した音声の質を評価(SDR, SIR, SARを算出)
def audio_eval(sample_rate, target_audio_path, interference_audio_path, mixed_audio_path, estimated_audio_path):
    """
    bss_eval_sourcesとbss_eval_imagesに関しては
    「http://bass-db.gforge.inria.fr/bss_eval/」
    を参照
    参考にしたソースコードは「https://github.com/craffel/mir_eval」
    """
    from .audio_evaluation.separation import bss_eval_sources, bss_eval_images

    # target = load_audio_file(target_audio_path, audio_length, sample_rate)[np.newaxis, :]
    # interference = load_audio_file(interference_audio_path, audio_length, sample_rate)[np.newaxis, :]
    # mixed = load_audio_file(mixed_audio_path, audio_length, sample_rate)[np.newaxis, :]
    # estimated = load_audio_file(estimated_audio_path, audio_length, sample_rate)[np.newaxis, :]

    # 音声データの読み込み
    target = sf.read(target_audio_path)[0][np.newaxis, :]
    interference = sf.read(interference_audio_path)[0][np.newaxis, :]
    mixed = sf.read(mixed_audio_path)[0][np.newaxis, :]
    estimated = sf.read(estimated_audio_path)[0][np.newaxis, :]
    
    # 各音声の長さの最大値を取得（評価時に音声の長さを揃える必要があるため）
    max_audio_length = np.amax(np.array([target.shape[1], interference.shape[1], mixed.shape[1], estimated.shape[1]]))
  
    # データが三次元の時、(手前, 奥), (上,下), (左, 右)の順番でパディングを実行
    target = np.pad(target, [(0, 0), (0, max_audio_length - target.shape[1]), (0, 0)], 'constant')
    interference = np.pad(interference, [(0, 0), (0, max_audio_length - interference.shape[1]), (0, 0)], 'constant') 
    mixed = np.pad(mixed, [(0, 0), (0, max_audio_length - mixed.shape[1]), (0, 0)], 'constant') 
    estimated = np.pad(estimated, [(0, 0), (0, max_audio_length - estimated.shape[1]), (0, 0)], 'constant') 

    reference = np.concatenate([target, interference], 0) # 目的音と外的雑音を結合する
    mixed = np.concatenate([mixed, mixed], 0) # referenceと同じ形になるように結合
    estimated = np.concatenate([estimated, estimated], 0) # referenceと同じ形になるように結合

    # シングルチャンネル用 (シングルチャンネルの場合音声はshape:[1, num_samples]の形式)
    if target.ndim == 2:
        mixed_result = bss_eval_sources(reference, mixed) # 混合音声のSDR, SIR, SARを算出
        reference_result = bss_eval_sources(reference, estimated) # モデルが推定した音声のSDR, SIR, SARを算出
        # 混合音声の評価結果
        sdr_mix = mixed_result[0][0]
        sir_mix = mixed_result[1][0]
        sar_mix = mixed_result[2][0]
        # 推定音声の評価結果
        sdr_est = reference_result[0][0]
        sir_est = reference_result[1][0]
        sar_est = reference_result[2][0]

    # マルチチャンネル用 (マルチチャンネルの場合音声はshape:[1, num_samples, num_channels]の形式)
    elif target.ndim == 3:
        mixed_result = bss_eval_images(reference, mixed) # 混合音声のSDR, SIR, SARを算出
        reference_result = bss_eval_images(reference, estimated) # モデルが推定した音声のSDR, SIR, SARを算出
        # 混合音声の評価結果
        sdr_mix = mixed_result[0][0]
        sir_mix = mixed_result[2][0]
        sar_mix = mixed_result[3][0]
        # 推定音声の評価結果
        sdr_est = reference_result[0][0]
        sir_est = reference_result[2][0]
        sar_est = reference_result[3][0]
    else:
        print("The number of audio channels is incorrect")

    return sdr_mix, sir_mix, sar_mix, sdr_est, sir_est, sar_est

# 正解ラベルのテキストと音声認識結果のテキストの距離を算出
def editDistance(r, h):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.

    Main algorithm used is dynamic programming.

    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d

def getStepList(r, h, d):
    '''
    This function is to get the list of steps in the process of dynamic programming.

    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and r.
    '''
    x = len(r)
    y = len(h)
    list = []
    while True:
        if x == 0 and y == 0: 
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] and r[x-1] == h[y-1]: 
            list.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y-1]+1:
            list.append("i")
            x = x
            y = y - 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1]+1:
            list.append("s")
            x = x - 1
            y = y - 1
        else:
            list.append("d")
            x = x - 1
            y = y
    return list[::-1]

# 音声認識評価結果をファイルに書き込む
def alignedPrint(list, r, h, result, result_path):
    '''
    This funcition is to print the result of comparing reference and hypothesis sentences in an aligned way.
    
    Attributes:
        list   -> the list of steps.
        r      -> the list of words produced by splitting reference sentence.
        h      -> the list of words produced by splitting hypothesis sentence.
        result -> the rate calculated based on edit distance.
    '''
    # 結果を保存するファイルの中身を初期化
    if os.path.exists(result_path):
        os.remove(result_path)
    
#     print("REF:", end=" ")
    with open(result_path, mode='a') as f:
        print("REF:", end=" ", file=f)
    for i in range(len(list)):
        if list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
#             print(" "*(len(h[index])), end=" ")
            with open(result_path, mode='a') as f:
                print(" "*(len(h[index])), end=" ", file=f)
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) < len(h[index2]):
#                 print(r[index1] + " " * (len(h[index2])-len(r[index1])), end=" ")
                with open(result_path, mode='a') as f:
                    print(r[index1] + " " * (len(h[index2])-len(r[index1])), end=" ", file=f)
            else:
#                 print(r[index1], end=" "),
                with open(result_path, mode='a') as f:
                    print(r[index1], end=" ", file=f)
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
#             print(r[index], end=" "),
            with open(result_path, mode='a') as f:
                print(r[index], end=" ", file=f)
#     print("\nHYP:", end=" ")
    with open(result_path, mode='a') as f:
        print("\nHYP:", end=" ", file=f)
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
#             print(" " * (len(r[index])), end=" ")
            with open(result_path, mode='a') as f:
                print(" " * (len(r[index])), end=" ", file=f)
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
#                 print(h[index2] + " " * (len(r[index1])-len(h[index2])), end=" ")
                with open(result_path, mode='a') as f:
                    print(h[index2] + " " * (len(r[index1])-len(h[index2])), end=" ", file=f)
            else:
#                 print(h[index2], end=" ")
                with open(result_path, mode='a') as f:
                    print(h[index2], end=" ", file=f)
        else:
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
#             print(h[index], end=" ")
            with open(result_path, mode='a') as f:
                print(h[index], end=" ", file=f)
#     print("\nEVA:", end=" ")
    with open(result_path, mode='a') as f:
        print("\nEVA:", end=" ", file=f)
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
#             print("D" + " " * (len(r[index])-1), end=" ")
            with open(result_path, mode='a') as f:
                print("D" + " " * (len(r[index])-1), end=" ", file=f)
        elif list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
#             print("I" + " " * (len(h[index])-1), end=" ")
            with open(result_path, mode='a') as f:
                print("I" + " " * (len(h[index])-1), end=" ", file=f)
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
#                 print("S" + " " * (len(r[index1])-1), end=" ")
                with open(result_path, mode='a') as f:
                    print("S" + " " * (len(r[index1])-1), end=" ", file=f)
            else:
#                 print("S" + " " * (len(h[index2])-1), end=" ")
                with open(result_path, mode='a') as f:
                    print("S" + " " * (len(h[index2])-1), end=" ", file=f)
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
#             print(" " * (len(r[index])), end=" ")
            with open(result_path, mode='a') as f:
                print(" " * (len(r[index])), end=" ", file=f)
#     print("\nWER: " + result)
    with open(result_path, mode='a') as f:
        print("\nWER: " + result, file=f)
    
# 音声認識性能（Word Error Rate; WER）の評価
def asr_eval(ref_text, hyp_text, result_path):
    """
    ref_text: 正解ラベルのテキスト （例） ['IT', 'IS', 'MARVELLOUS']
    hyp_text: 音声認識結果のテキスト （例） ['IT', 'WAS', 'MADNESS']
    result_path: 音声認識性能の評価結果を保存するファイルのパス
    """
    # build the matrix
    d = editDistance(ref_text, hyp_text)

    # find out the manipulation steps
    list = getStepList(ref_text, hyp_text, d)

    # print the result in aligned way
    result = float(d[len(ref_text)][len(hyp_text)]) / len(ref_text) * 100
    result_str = str("%.2f" % result) + "%"
    alignedPrint(list, ref_text, hyp_text, result_str, result_path)
    return result