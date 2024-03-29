# U-TasNet-Beam

## Abstruct

PyTorch implementation of "Adaptation of robots to the real environment by simultaneous execution of
 dereverberation, denoising and speaker separation using neural beamformer".



![](./utils/assets/U-TasNet-Beam.png)



## Dependencies

This code was tested on Python 3.8.1 with PyTorch 1.10.0, torchvision 0.11.1 and torchaudio 0.10.0. Optionally Install espnet and espnet-model-zoo if you need it.

```
$ pip3 install -r requirements.txt
```



## Prepare Dataset

1. Download Noisy Speech Database

   Get Noisy Speech Database at https://datashare.ed.ac.uk/handle/10283/2791.

   Please download the following.

   - `clean_testset_wav.zip`
   - `clean_trainset_28spk_wav.zip`
   - `noisy_testset_wav.zip`
   - `noisy_trainset_28spk_wav.zip`
   - `testset_txt.zip`

2. Spatialize audio by convolving RIR (Room Impulse Response)

   First, unzip zip file to desired folder.

   ```
   $ tar *.zip 
   ```

   Second, open jupyter notebook and run `make_train_val_datasets_MCCUNet.ipynb` to make a training and validation dataset for MCCU-Net.
   Besides, run `make_train_val_datasets_MCConvTasNet.ipynb` to make a training and validation dataset for MCConvTasNet.


   ```
   $ jupyter notebook
   ```

   Finally, run `make_test_datasets.ipynb` to make a test dataset for performace evaluation of U-TasNet-Beam.





## Training Multi-channel Complex U-Net

```
$ python3 training_MCComplexUnet.py
```



## Training Multi-channel Conv-TasNet

```
$ python3 training_MCConvTasNet.py
```





## Inference and evaluation

1. Download pretrained model for speaker recognition system

   This method utilizes speaker recognition system ([d-vector embeddings](https://google.github.io/speaker-id/publications/GE2E/)).

   Get pretrained model for speaker recognition system at [this GDrive link](https://drive.google.com/file/d/1YFmhmUok-W76JkrfA0fzQt3c-ZsfiwfL/view?usp=sharing).

   This model was trained with [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) dataset,
   where utterances are randomly fit to time length [70, 90] frames.
   Tests are done with window 80 / hop 40 and have shown equal error rate about 1%.
   Data used for test were selected from first 8 speakers of [VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) test dataset, where 10 utterances per each speakers are randomly selected.

   **Update**: Evaluation on VoxCeleb1 selected pair showed 7.4% EER.

2. Run

   ```
   $ python3 inference.py
   ```

   **Option**

   - `-sr` : sampling rate (Default 16000)
   - `-bl` : batch size of mask estimator and beamformer input (Default 48000)
   - `-c` : number of audio channels (Default 8)
   - `-dmt` : denoising and dereverberation model type
   - `-ssmt` : speaker separation model type
   - `-bt` : beamformer type
   
   If you evaluate the performance by using multiple audio data at once, use `evaluate_neural_beamformer.ipynb`.




## Online demo

1. Prepare the microphone array

   You can use TAMAGO-03 microphone array with 8 microphones.

2. Run

   Open two terminals and run following commands in each terminal (Mac or Linux). Be careful to set the Julius server URL in `RealTimeDemo.py` and `speech_extracter_interface.py` to the correct one.
   
   - Server

     ```
     $ python3 asr_server_julius.py
     ```
   
   - Client
   
     ```
     $ python3 RealTimeDemo.py -em -d 0 -mg 20
     ```
   
     **Option**
   
     - `-em` : Whether model extracts audio or not
   
     - `-d` : Input device (numeric ID or substring) (you can check ID by running following commands)
        ```
        $ python3 
        >>> import sounddevice
        >>> sounddevice.query_devices()
        ```
   
     - `-mg`: Increase microphone gain

   If you can use g++ complier on linux, open three terminals and run following commands in each terminal (Input stream speed is faster).

   - ASR server

     ```
     $ python3 asr_server_julius.py
     ```
   
   - Speech extracter interface (server & client)

     ```
     $ python3 speech_extracter_interface.py -em -mg 20
     ```
   
   - Input stream client
   
     ```
     $ g++ mic_record_to_speech_extracter.cpp -lasound -lm -o mic_record_to_speech_extracter
     $ ./mic_record_to_speech_extracter plughw:2,0
     ```
     If you run `arecord -l` and the following is displayed, specify the argument part as `plughw:[card number],[subdevice number]` 
     ```
      card 2: TAMAGO03 [TAMAGO-03], device 0: USB Audio [USB Audio]
      Subdevices: 1/1
      Subdevice #0: subdevice #0
     ```







## Author

Daichi Nagano at nakazawa lab

E-mail: naganod1@gmail.com



