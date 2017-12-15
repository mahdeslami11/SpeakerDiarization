# SpeakerDiarization
## Requirements:
I have used Anaconda distribution with python3. 
### Install other dependencies as follows:
pip install pyannote.metrics librosa
## Main File:
Use ReDiarization.py to run the speaker Diarization on audio files.
See the "main" carefully to set the flags to run specific scenarios.
e.g. Either to use sparse feature extraction with MFCC etc.
Set your own paths in "main" for input audio and other output files. 
## Data Set:
Short audio file consisting of 40seconds recording with annotation is placed in "dataset" folder. To use large audio files, you can use the AMI corpus with annotation from http://groups.inf.ed.ac.uk/ami/download/.
## References:
Main implementation of Speaker Diarization based upon GMM Heirarchical Agglomerative clustering is taken from [1], while "Speech Activity detection" is taken from [2]. Used pyannote.metrics to compute Diarization Error rate, Cluster purity etc from [3].

[1] E. Gonina, G. Friedland, H. Cook and K. Keutzer, "Fast speaker diarization using a high-level scripting language," 2011 IEEE Workshop on Automatic Speech Recognition & Understanding, Waikoloa, HI, 2011, pp. 553-558.

[2] @article{giannakopoulos2015pyaudioanalysis,
  title={pyAudioAnalysis: An Open-Source Python Library for Audio Signal Analysis},
  author={Giannakopoulos, Theodoros},
  journal={PloS one},
  volume={10},
  number={12},
  year={2015},
  publisher={Public Library of Science}
}

[3] @inproceedings{pyannote.metrics,
  author = {Herv\'e Bredin},
  title = {{pyannote.metrics: a toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems}},
  booktitle = {{Interspeech 2017, 18th Annual Conference of the International Speech Communication Association}},
  year = {2017},
  month = {August},
  address = {Stockholm, Sweden},
  url = {http://pyannote.github.io/pyannote-metrics},
}
