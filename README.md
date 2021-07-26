# Raw Audio Transformer

We propose an architecture for an autoregressive transformer for raw audio synthesis. We condition this synthesis on pitch, instrument and amplitude scale per sample of the raw audio data using floating point values. Our primary aim is to be able to synthesize raw audio samples interpolating continuously either on pitch, amplitude or instrument. 

This repository was developed in part based on Peter Bloem's amazing blogpost and videos at http://peterbloem.nl/blog/transformers

## Dataset

Our architecture was trained on a pre-processed NSynth brass and reed instrument dataset. For training and inference, we choose an octave of pitches ranging from MIDI 64 (~330Hz) to MIDI 76 (~660 Hz) for both the instruments and extract the middle 2.5 seconds of raw audio sampled at a sampling rate of 16kHz. Each resulting `.wav` file is then scaled by an amplitude range of `[0, 1)` in steps of 0.1 to create 260 pre-processed audio files (13 pitches * 10 amplitude scales * 2 instruments). We use `paramManager` to manage the parameters on each individual audio file. Please see the repository https://github.com/lonce/paramManager for more details.

## Installation & Running

Please follow the steps below to run an instance of this project using either Docker or our University's Singularity container -  

Note: The current version of the repository uses GPU device (`cuda:0`) only.

### Docker  
  
1. Update the config/config.json `data_dir` to `/data/nsynth.64.76.dl`
2. Build docker container using ** - 
`docker image build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --file container-config/Dockerfile --tag audiotx:v1`
3. Run container using ** - 
`docker run  --shm-size=10g --gpus "device=0" -it -v $(pwd):/audiotx -v <params_and_audio_data_dir>:/data --rm audiotx:v1`    
4. Run main.py -  
`python main.py`  

### NUS Singularity Container   

1. Update the config/config.json `data_dir` to `<location of data on /hpctmp or /scratch>`   
2. Submit PBS job using ** -  
`qsub container-config/audiotx.pb`   


** Note: Please run the scripts from the root of the project directory