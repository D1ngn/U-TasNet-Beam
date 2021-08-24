#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Julius web server (with CherryPy:http://www.cherrypy.org/)
# written by Ryota NISHIMURA 2015/Dec./16

### configure ###########
JULIUS_HOME		= "/Users/nagano.daichi/julius/julius"
JULIUS_EXEC		= "./julius -C /Users/nagano.daichi/julius/dictation-kit-4.5/main.jconf -C /Users/nagano.daichi/julius/dictation-kit-4.5/am-gmm.jconf -nostrip -input file -outfile" # 「前に進め」、「後ろに下がれ」などを認識
# JULIUS_EXEC		= "./julius -C /Users/nagano.daichi/julius/dictation-kit-4.5/am-gmm.jconf -gram /Users/nagano.daichi/julius/dict/order -nostrip -input file -outfile"
# JULIUS_EXEC		= "./julius -C /Users/nagano.daichi/julius/dictation-kit-4.5/main.jconf -C /Users/nagano.daichi/julius/dictation-kit-4.5/am-dnn.jconf -dnnconf /Users/nagano.daichi/julius/dictation-kit-4.5/julius.dnnconf -nostrip -input file -outfile" # 自由な単語を認識（処理速度は遅い）
SERVER_PORT 	= 8000
ASR_FILEPATH	= '/Users/nagano.daichi/MaskBeamformer/recog_result/asr_result/'
ASR_IN			= 'ch_asr.wav'
ASR_RESULT		= 'ch_asr.out'
OUT_CHKNUM		= 5 # for avoiding that the output file is empty

### import ##############
import cherrypy
import subprocess
import sys
import os
import time
import socket
from cherrypy import request

import soundfile as sf
import numpy as np

from io import BytesIO

### class define ########
class ASRServer(object):
	# Julius execution -> subprocess
    p = subprocess.Popen (JULIUS_EXEC, shell=True, cwd=JULIUS_HOME, 
        stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
    (stdouterr, stdin) = (p.stdout, p.stdin)

    # main task
    def index(self):
        return """
        <html><body>
            <h2>Julius Server</h2>
                USAGE:<br />
                - 16000Hz, wav(or raw)-file, big-endian, mono<br />
                <br />
                <form action="asr_julius" method="post" enctype="multipart/form-data">
                filename: <input type="file" name="myFile" /><br />
                <input type="submit" />
                </form>
            </body></html>
            """
    index.exposed = True

    def asr_julius(self, myFile):
        # receive WAV file from client & write WAV file
        # クライアント側からバイナリデータ形式で送られてきたwavデータを一度wavファイルに保存する
        # with open(ASR_FILEPATH + ASR_IN, 'wb') as f:
        #     f.write(myFile.file.read())
        # f.close()
        audio_data = np.load(BytesIO(myFile.file.read()))
        sf.write(ASR_FILEPATH + ASR_IN, audio_data, 16000)

		# ASR using Julius
        if os.path.exists(ASR_FILEPATH + ASR_RESULT):
            os.remove(ASR_FILEPATH + ASR_RESULT) # delete a previous result file
        send_msg = ASR_FILEPATH + ASR_IN + '\n'
        self.p.stdin.write(send_msg.encode()) # send wav file name to Julius
        # self.p.stdin.write(ASR_FILEPATH + ASR_IN + '\n')	# send wav file name to Julius
        self.p.stdin.flush() # バッファの解放
        
        # wait for result file creation & result writing (avoid the file empty)
        while not (os.path.exists(ASR_FILEPATH + ASR_RESULT) and len(open(ASR_FILEPATH + ASR_RESULT).readlines()) == OUT_CHKNUM):
            time.sleep(0.1)
        
        # read result file & send it to client
        outlines = open(ASR_FILEPATH + ASR_RESULT).readline()[11:] # 認識結果のみ
        # outlines = open(ASR_FILEPATH + ASR_RESULT).read()
        # outlines = "<xmp>" + outlines + "</xmp>"
        return outlines
    asr_julius.exposed = True


if __name__ == "__main__":
    # get own IP
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    server_ip = s.getsockname()[0]

    # start the CherryPy server
    cherrypy.config.update({'server.socket_port': SERVER_PORT,})
    cherrypy.config.update({'server.socket_host': server_ip,})
    cherrypy.quickstart(ASRServer())