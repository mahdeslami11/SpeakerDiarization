# SpeakerDiarization
## Requirements:
I have used Anaconda distributino with python3. 
### Install other dependencies as follows:
pip install pyannote.metrics simplejson eyed3 pydub librosa
## Main File:
Use ReDiarization.py to run the speaker Diarization on audio files.
See the "main" carefully to set the flags to run specific scenarios.
e.g. Either to use sparse feature extraction with MFCC etc.
Set your own paths in "main" for input audio and other output files. 
## Data Set:
Short audio file consisting of 40seconds recording with annotation is placed in "dataset" folder. To use large audio files, you can use the AMI corpus with annotation from http://groups.inf.ed.ac.uk/ami/download/.
## References:
Main implementation taken from the first reference, while "Speech Activity detection" is taken from the second one.

[1] E. Gonina, G. Friedland, H. Cook and K. Keutzer, "Fast speaker diarization using a high-level scripting language," 2011 IEEE Workshop on Automatic Speech Recognition & Understanding, Waikoloa, HI, 2011, pp. 553-558.

[2] Giannakopoulos T (2015) pyAudioAnalysis: An Open-Source Python Library for Audio Signal Analysis. PLoS ONE 10(12): e0144610.
