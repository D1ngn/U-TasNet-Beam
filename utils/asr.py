
import os
import subprocess
import soundfile as sf  


# ESPNetを用いた音声認識
class ASR():
    def __init__(self, lang='eng'):
        # 必要モジュールをインポート（あらかじめ「pip3 install espnet_model_zoo」を実行）
        from espnet_model_zoo.downloader import ModelDownloader
        from espnet2.bin.asr_inference import Speech2Text
        # E2E-ASRモデルのインスタンスを作成
        d = ModelDownloader()
        # 英語版
        if lang == 'eng':
            self.speech2text = Speech2Text(
                # タスク（音声認識）と使用するコーパスを指定し、学習済みモデルをダウンロード
                **d.download_and_unpack(task="asr", corpus="librispeech")
            )
        # 日本語版
        elif lang == 'jp':
            self.speech2text = Speech2Text(
                # タスク（音声認識）と使用するコーパスを指定し、学習済みモデルをダウンロード
                **d.download_and_unpack(task="asr", corpus="jsut")
            )
    # 音声認識を実行
    def speech_recognition(self, audio_path):
        audio_data, _ = sf.read(audio_path)
        text, token, *_ = self.speech2text(audio_data)[0]
        return text

# Juliusを用いた音声認識
def asr_julius(input_file_path):
    temp_file = "julius_asr_recog_result.txt"
    # juliusによる音声認識を実行し、結果をファイルに出力
    # # 混合ガウスモデル（GMM）ベースの音響モデルを用いる場合→今は「前に進め」、「後ろに退がれ」など（オリジナルの単語辞書に登録されたもの）を認識
    # asr_cmd = "echo {} | julius -C ~/julius/dictation-kit-4.5/main.jconf -C ~/julius/dictation-kit-4.5/am-gmm.jconf -nostrip -input rawfile -quiet > {}".format(input_file_path, temp_file)
    # DNNベースの音響モデルを用いる場合→今はさまざまな日本語を認識（英語は不可）
    asr_cmd = "echo {} | julius -C ~/julius/dictation-kit-4.5/main.jconf -C ~/julius/dictation-kit-4.5/am-dnn.jconf -dnnconf ~/julius/dictation-kit-4.5/julius.dnnconf -nostrip -input rawfile -quiet > {}".format(input_file_path, temp_file)
    subprocess.call(asr_cmd, shell=True)
    # 出力ファイルから認識結果の部分のみを抽出
    with open(temp_file) as f:
        lines = f.readlines()
    recog_text_line = [line.strip() for line in lines if line.startswith('sentence1')] # "sentence1"から始まる行をサーチ
    recog_result = recog_text_line[0][12:-2] # "sentence1: "から"。"の間の文章を抽出
    # 余分なファイルが残らないように削除
    os.remove(temp_file)
    return recog_result