/*
参考：「https://gist.github.com/kazz12211/ba3989e74fd76231046c1a5d95651b5e」
*/

#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <alsa/asoundlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

using namespace std;

int main(int argc, char *argv[]) {
	// 録音（キャプチャ）用の設定
	int i;
	int buffer_size;
	int err;
	char *buffer;
	int buffer_frames = 128;
	unsigned int sample_rate = 16000;
	unsigned int channel = 8;
	int audio_length = 3; // second
	snd_pcm_t *capture_handle;
	snd_pcm_hw_params_t *hw_params;
	snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE; // Signed 16 bit Little Endian
	mode_t dir_mode; // 音声保存用ディレクトリを作成する際のモード
	dir_mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
	// ソケット通信用の設定
	int sockfd;
    struct sockaddr_in addr;
    char receive_data[30]; // 受信データ
	// ソケット生成
	// AF_INETはIPv4、SOCK_STREAMはTCPであることを表す
    if((sockfd = socket( AF_INET, SOCK_STREAM, 0)) < 0){
        perror("socket");
    }
	// 送信先アドレス・ポート番号設定
    addr.sin_family = AF_INET;
    addr.sin_port = htons(1234);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
	// サーバ接続
    connect(sockfd, (struct sockaddr *)&addr, sizeof(struct sockaddr_in));
	
	/**
	 * PCMを開く
	 * 引数：
	 * PCMハンドル (snd_pcm_t**)
	 * PCMハンドルの識別子 (const char *)
	 * キャプチャストリーム (snd_pcm_stream_t)
	 * オープンモード
	 *
	 **/
	if((err = snd_pcm_open(&capture_handle, argv[1], SND_PCM_STREAM_CAPTURE, 0)) < 0) {
		fprintf(stderr, "cannot open audio device %s (%s)\n", argv[1], snd_strerror(err));
		exit(1);
	}
	// mallocを使ってハードウェアパラメーターのコンフィグレーション領域を確保する
	if((err = snd_pcm_hw_params_malloc(&hw_params)) < 0) {
		fprintf(stderr, "cannot allocate hardware parameter structure (%s)\n", snd_strerror(err));
		exit(1);
	}
	//コンフィグレーション領域を初期化する
	if((err = snd_pcm_hw_params_any(capture_handle, hw_params)) < 0) {
		fprintf(stderr, "cannot initialize hardware parameter structure (%s)\n", snd_strerror(err));
		exit(1);
	}
	/*
	 * コンフィグレーション領域に１つのアクセスタイプに設定する
	 * 引数：
	 * PCMハンドル (snd_pcm_t *)
	 * コンフィグレーション領域 (snd_pcm_hw_params *)
	 * アクセスタイプ (snd_pcm_access_t)
	 *
	 * SND_PCM_ACCESS_RW_INTERLEAVED = snd_pcm_readi/snd_pcm_writei access
	 */
	if((err = snd_pcm_hw_params_set_access(capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
		fprintf(stderr, "cannot set access type (%s)\n", snd_strerror(err));
		exit(1);
	}
	/*
	 * コンフィグレーション領域に１つのサンプリングフォーマットに設定する
	 * 引数
	 * PCMハンドル (snd_pcm_t *)
	 * コンフィグレーション領域 (snd_pcm_hw_params *)
	 * サンプリングフォーマット (snd_pcm_format_t)
	 */
	if((err = snd_pcm_hw_params_set_format(capture_handle, hw_params, format)) < 0) {
		fprintf(stderr, "cannot set sample format (%s)\n", snd_strerror(err));
		exit(1);
	}
	/*
	 * コンフィグレーション領域にターゲットに近いサンプリングレートに設定する
	 * 引数
	 * PCMハンドル (snd_pcm_t *)
	 * コンフィグレーション領域 (snd_pcm_hw_params *)
	 * 大凡のサンプリングレート (unsigned int *)
	 * サブユニットの方向
	 */
	if((err = snd_pcm_hw_params_set_rate_near(capture_handle, hw_params, &sample_rate, 0)) < 0) {
		fprintf(stderr, "cannot set sample rate (%s)\n", snd_strerror(err));
		exit(1);
	}
	/*
	 * コンフィグレーション領域にチャネル数を設定する
	 * 引数
	 * PCMハンドル (snd_pcm_t *)
	 * コンフィグレーション領域 (snd_pcm_hw_params *)
	 * チャネル数 (unsigned int)
	 */
	if((err = snd_pcm_hw_params_set_channels(capture_handle, hw_params, channel)) < 0) {
		fprintf(stderr, "cannot set channel count (%s)\n", snd_strerror(err));
		exit(1);
	}
	//コンフィグレーションを設定
	if((err = snd_pcm_hw_params(capture_handle, hw_params)) < 0) {
		fprintf(stderr, "cannot set hw_params (%s)\n", snd_strerror(err));
		exit(1);
	}
	//コンフィグレーション領域の開放
	snd_pcm_hw_params_free(hw_params);
	// オーディオインターフェースの開始
	if((err = snd_pcm_prepare(capture_handle)) < 0) {
		fprintf(stderr, "cannot prepare audio interface for use (%s)", snd_strerror(err));
		exit(1);
	}

	// キャプチャー用のバッファーのサイズを算出
	buffer_size = buffer_frames * snd_pcm_format_width(format)/8 * channel; /* 2 bytes/sample, 8 channels */
	// 録音データを一定時間ごとに区切り、speech extracterモジュール（提案システム）に送り続ける
	while(true){
		// 音声保存用ファイルの名前につける日付の設定
		time_t now = time(nullptr); // 現在時刻（1970年1月1日からの経過秒）
		//形式を変換する    
		tm* lt = localtime(&now);
		stringstream ss;
		ss<<"20"<<lt->tm_year-100; //100を引くことで20xxのxxの部分になる
		ss<<"-"<<lt->tm_mon+1; //月を0からカウントしているため
		ss<<"-"<<lt->tm_mday; //そのまま
		ss<<"-"<<lt->tm_hour; //そのまま
		ss<<"-"<<lt->tm_min; //そのまま
		ss<<"-"<<lt->tm_sec; //そのまま 
		string nowtime = ss.str(); // （例）nowtime = "2021-8-30-14-44-53"
		// ファイル名を現在時刻に設定した音声ファイルのパスを生成（処理中にファイルの上書きを避けるため）
		char raw_path[] = "./temp/";
		mkdir(raw_path, dir_mode);
		strcat(raw_path, nowtime.c_str());
		strcat(raw_path, ".raw");
		FILE* fp;
		fp = fopen(raw_path, "wb");
		// キャプチャー用のバッファーを確保
		buffer = (char*) malloc(buffer_size);
		// キャプチャー
		for(i = 0; i < sample_rate*audio_length/buffer_frames; i++) {
			if((err = snd_pcm_readi(capture_handle, buffer, buffer_frames)) != buffer_frames) {
				fprintf(stderr, "read from audio interface failed (%s)\n", snd_strerror(err));
				exit(1);
			}
			// 音声ファイルに保存
			fwrite(buffer, sizeof(char), buffer_size, fp);
		}
		//音声ファイルパスをサーバに送信
		send(sockfd, raw_path, strlen(raw_path), 0);
		// データ受信
		// サーバーからの返答を受信
		recv(sockfd, receive_data, 30, 0);
		// 受信データを標準出力
		cout << receive_data << endl;
		// メモリの開放
		if (buffer != NULL) {
			free(buffer);
		}
		// ファイルを閉じる
		fclose(fp);
	}
	// ソケットを閉じる
	close(sockfd);
	// ストリームを閉じる
	if (capture_handle != NULL) {
		snd_pcm_close(capture_handle);
	}
	exit(0);
  	return 0;
}