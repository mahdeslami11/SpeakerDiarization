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
Main implementation of Speaker Diarization is based on GMM Heirarchical Agglomerative clustering which is taken from [1], while "Speech Activity detection" is taken from [2]. Used pyannote.metrics to compute Diarization Error rate, Cluster purity etc. from [3].

__[1] E. Gonina, G. Friedland, H. Cook and K. Keutzer, "Fast speaker diarization using a high-level scripting language," 2011 IEEE Workshop on Automatic Speech Recognition & Understanding, Waikoloa, HI, 2011, pp. 553-558.__

__[2] @article{giannakopoulos2015pyaudioanalysis,
  title={pyAudioAnalysis: An Open-Source Python Library for Audio Signal Analysis},
  author={Giannakopoulos, Theodoros},
  journal={PloS one},
  volume={10},
  number={12},
  year={2015},
  publisher={Public Library of Science}
}__

__[3] @inproceedings{pyannote.metrics,
  author = {Herv\'e Bredin},
  title = {{pyannote.metrics: a toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems}},
  booktitle = {{Interspeech 2017, 18th Annual Conference of the International Speech Communication Association}},
  year = {2017},
  month = {August},
  address = {Stockholm, Sweden},
  url = {http://pyannote.github.io/pyannote-metrics},
}__



1. Target & codes function


this project is about speaker diarization. it means we have meetings, conversations, and... then there are more than 2 people that they are talking. the detector recognize which voice from who is playing and emphsize them one by one without over lapping.
after all we use video detector and voice detector at same time and we delete the muted voice to recognizing better. we use GMM cluster and SAD that you can find in files to run the ReDiarization.py.

2.Describe innovation

Speakers diarization with detecting the latency and overlap of online speakers
The project has been based on MIT databases for training network. 
In order to compile the desired code, the following stages must be taken:
1.install pycharm
2.install pip python --upgrade
the python version =3.9 (it is necessary)
3. pip install pip --upgrade
4. install anaconda (it takes time)
after the anaconda is installed the environment in conda based on python 3.9 is made.
5.the below libraries should be add in order to use them in code.
diart, tourch, tourchaudio, pyannote.audio 
6.conda create -n diart python=3.9
7.conda activate diart
8.conda create -n diart python=3.8
9.conda activate diart
10.pip install diart
11.pyannote.audio should be installed
12. the terms of MIT have to be consented by user by following the 
 https://huggingface.co/pyannote/segmentation
13. order the token for login open the below link:
14. pip install huggingface_hub
15.conda install -c conda-forge huggingface_hub
16. use the token code for logging to the MIT database
17.the main.py must be run
18. please enter the token number 
19. after compile the main.py
20. the script turn the laptop's speaker on and the voice of speaker are saved every 0.5 in buffers.
21. by using pipeline all voices are recorded and according to the frequency the speaker voices is recognized.
22. the network will be trained by 5 sec recording 
23. the crucial parts of this project are latency and the over-lap of speakers' voices.
   the problems are solved by separating the frequency according to the trained network and pytourch library.


3.Change in source codes

In oreder to compile the desired code the following stages must be taken:
1.install pycharm
2.install pip python --upgrade
the python version =3.8 ( it is necessary)
3. pip instal pip --upgrade
4. inistall anaconda ( it takes time)
after the anaconda is inistalled the environment in conda based on python 3.8 is made.
5.the below libraries should be add in order to use them in code.
diart,tourch,tourchaudio,py
6.conda create -n diart python=3.9
7.conda activate diart
8.conda create -n diart python=3.8
9.conda activate diart
10.pip install diart
11.pyannote.audio should be installed
12. the terms of MIT have to be consented by user by following the 
 https://huggingface.co/pyannote/segmentation
13. order the token for login open the below link:
14. pip install huggingface_hub
15.conda install -c conda-forge huggingface_hub
16. use the token code for logging to the MIT database
17.the main.py must be run
18. please enter the token number 
19. after compile the main.py
20. the script turn the laptop's speaker on and the voice of speaker are saved every 0.5 in buffers.
21. by using pipeline all voices are recorded and according to the frequency the speaker voices is regognize.
22. the network will be trained by 5 sec recording 
23. the crucial part of this project is 



4. Result

import asyncio
import base64
from pathlib import Path
from queue import SimpleQueue
from typing import Text, Optional, AnyStr

import numpy as np
import sounddevice as sd
import torch
import websockets
from einops import rearrange
from rx.subject import Subject
from torchaudio.io import StreamReader
import signal

from .audio import FilePath, AudioLoader


class AudioSource:
    """Represents a source of audio that can start streaming via the `stream` property.

    Parameters
    ----------
    uri: Text
        Unique identifier of the audio source.
    sample_rate: int
        Sample rate of the audio source.
    """
    def __init__(self, uri: Text, sample_rate: int):
        self.uri = uri
        self.sample_rate = sample_rate
        self.stream = Subject()

    @property
    def duration(self) -> Optional[float]:
        """The duration of the stream if known. Defaults to None (unknown duration)."""
        return None

    def read(self):
        """Start reading the source and yielding samples through the stream."""
        raise NotImplementedError

    def close(self):
        """Stop reading the source and close all open streams."""
        raise NotImplementedError


class FileAudioSource(AudioSource):
    """Represents an audio source tied to a file.

    Parameters
    ----------
    file: FilePath
        Path to the file to stream.
    sample_rate: int
        Sample rate of the chunks emitted.
    """
    def __init__(
        self,
        file: FilePath,
        sample_rate: int,
        padding_end: float = 0,
        block_size: int = 1000,
    ):
        super().__init__(Path(file).stem, sample_rate)
        self.loader = AudioLoader(self.sample_rate, mono=True)
        self._duration = self.loader.get_duration(file)
        self.file = file
        self.resolution = 1 / self.sample_rate
        self.block_size = block_size
        self.padding_end = padding_end
        self.is_closed = False

    @property
    def duration(self) -> Optional[float]:
        # The duration of a file is known
        return self._duration + self.padding_end

    def read(self):
        """Send each chunk of samples through the stream"""
        waveform = self.loader.load(self.file)

        # Add zero padding at the end if required
        if self.padding_end > 0:
            num_pad_samples = int(np.rint(self.padding_end * self.sample_rate))
            zero_padding = torch.zeros(waveform.shape[0], num_pad_samples)
            waveform = torch.cat([waveform, zero_padding], dim=1)

        # Split into blocks
        _, num_samples = waveform.shape
        chunks = rearrange(
            waveform.unfold(1, self.block_size, self.block_size),
            "channel chunk sample -> chunk channel sample",
        ).numpy()

        # Add last incomplete chunk with padding
        if num_samples % self.block_size != 0:
            last_chunk = waveform[:, chunks.shape[0] * self.block_size:].unsqueeze(0).numpy()
            diff_samples = self.block_size - last_chunk.shape[-1]
            last_chunk = np.concatenate([last_chunk, np.zeros((1, 1, diff_samples))], axis=-1)
            chunks = np.vstack([chunks, last_chunk])

        # Stream blocks
        for i, waveform in enumerate(chunks):
            try:
                if self.is_closed:
                    break
                self.stream.on_next(waveform)
            except BaseException as e:
                self.stream.on_error(e)
                break
        self.stream.on_completed()
        self.close()

    def close(self):
        self.is_closed = True


class MicrophoneAudioSource(AudioSource):
    """Represents an audio source tied to the default microphone available"""

    def __init__(self, sample_rate: int, block_size: int = 1000):
        super().__init__("live_recording", sample_rate)
        self.block_size = block_size
        self._mic_stream = sd.InputStream(
            channels=1,
            samplerate=sample_rate,
            latency=0,
            blocksize=self.block_size,
            callback=self._read_callback
        )
        self._queue = SimpleQueue()

    def _read_callback(self, samples, *args):
        self._queue.put_nowait(samples[:, [0]].T)

    def read(self):
        self._mic_stream.start()
        while self._mic_stream:
            try:
                while self._queue.empty():
                    if self._mic_stream.closed:
                        break
                self.stream.on_next(self._queue.get_nowait())
            except BaseException as e:
                self.stream.on_error(e)
                break
        self.stream.on_completed()
        self.close()

    def close(self):
        self._mic_stream.stop()
        self._mic_stream.close()


class WebSocketAudioSource(AudioSource):
    """Represents a source of audio coming from the network using the WebSocket protocol.

    Parameters
    ----------
    sample_rate: int
        Sample rate of the chunks emitted.
    host: Text | None
        The host to run the websocket server. Defaults to ``None`` (all interfaces).
    port: int
        The port to run the websocket server. Defaults to 7007.
    """
    def __init__(self, sample_rate: int, host: Optional[Text] = None, port: int = 7007):
        name = host if host is not None and host else "localhost"
        uri = f"{name}:{port}"
        # FIXME sample_rate is not being used, this can be confusing and lead to incompatibilities.
        #  I would prefer the client to send a JSON with data and sample rate, then resample if needed
        super().__init__(uri, sample_rate)
        self.host = host
        self.port = port
        self.websocket = None
        self.stop = None

    async def _ws_handler(self, websocket):
        self.websocket = websocket
        try:
            async for message in websocket:
                # Decode chunk encoded in base64
                byte_samples = base64.decodebytes(message.encode("utf-8"))
                # Recover array from bytes
                samples = np.frombuffer(byte_samples, dtype=np.float32)
                # Reshape and send through
                self.stream.on_next(samples.reshape(1, -1))
            self.stream.on_completed()
            self.close()
        except websockets.ConnectionClosedError as e:
            self.stream.on_error(e)

    async def _async_read(self):
        loop = asyncio.get_running_loop()
        self.stop = loop.create_future()
        loop.add_signal_handler(signal.SIGTERM, self.stop.set_result, None)
        async with websockets.serve(self._ws_handler, self.host, self.port):
            await self.stop

    async def _async_send(self, message: AnyStr):
        await self.websocket.send(message)

    def read(self):
        """Starts running the websocket server and listening for audio chunks"""
        asyncio.run(self._async_read())

    def close(self):
        if self.websocket is not None:
            # The value could be anything
            self.stop.set_result(True)

    def send(self, message: AnyStr):
        """Send a message through the current websocket.

        Parameters
        ----------
        message: AnyStr
            Bytes or string to send.
        """
        # A running loop must exist in order to send back a message
        ws_closed = "Websocket isn't open, try calling `read()` first"
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError(ws_closed)

        if not loop.is_running():
            raise RuntimeError(ws_closed)

        # TODO support broadcasting to many clients
        # Schedule a coroutine to send back the message
        if message:
            asyncio.run_coroutine_threadsafe(self._async_send(message), loop=loop)


class TorchStreamAudioSource(AudioSource):
    def __init__(
        self,
        uri: Text,
        sample_rate: int,
        streamer: StreamReader,
        stream_index: Optional[int] = None,
        block_size: int = 1000,
    ):
        super().__init__(uri, sample_rate)
        self.block_size = block_size
        self._streamer = streamer
        self._streamer.add_basic_audio_stream(
            frames_per_chunk=self.block_size,
            stream_index=stream_index,
            format="fltp",
            sample_rate=self.sample_rate,
        )
        self.is_closed = False

    def read(self):
        for item in self._streamer.stream():
            try:
                if self.is_closed:
                    break
                # shape (samples, channels) to (1, samples)
                chunk = np.mean(item[0].numpy(), axis=1, keepdims=True).T
                self.stream.on_next(chunk)
            except BaseException as e:
                self.stream.on_error(e)
                break
        self.stream.on_completed()
        self.close()

    def close(self):
        self.is_closed = True
        
        
        
5. Original Paper

you can find the orginal file of paper in the files section. I uploaded there.

6. Interduce myself

This is Narges Pourakhlaghi, Master student of Biomedical engineering, Bioelectric branch in Islamic Azad University South Tehran Branch
this project belongs to DSP- Professor Dr. Mehdi Eslami
first semester 1401

7. Updated file

you can find them in the files.


8. Explaining video

https://drive.google.com/file/d/1WM5v1-DJ5ZNK8mc_X8YUzLWRRy-BhOkp/view?usp=share_link


9. Proposal

you can find it in the files.
11.ز
