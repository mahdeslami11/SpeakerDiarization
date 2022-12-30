
from huggingface_hub import HfApi

import warnings
##/////////////////////////
from huggingface_hub import hf_hub_download
from huggingface_hub import login

hf_hub_download(
repo_id="google/pegasus-xsum",
filename="config.json",
revision="4d33b01d79672f27f001f6abade33f22d993b151")

login()

api = HfApi()
api.create_repo(repo_id="super-cool-model")


from diart import OnlineSpeakerDiarization
from diart.sources import MicrophoneAudioSource
from diart.inference import RealTimeInference
from diart.sinks import RTTMWriter
from diart import OnlineSpeakerDiarization
from diart.sources import MicrophoneAudioSource
from diart.inference import RealTimeInference
from diart.sinks import RTTMWriter
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="ACCESS_TOKEN_GOES_HERE")


diarization = pipeline("audio.wav")


with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)

import torch
print(torch.__version__)


pipeline = OnlineSpeakerDiarization()
mic = MicrophoneAudioSource(pipeline.config.sample_rate)
inference = RealTimeInference(pipeline, mic, do_plot=True)
inference.attach_observers(RTTMWriter(mic.uri, "/output/file.rttm"))
prediction = inference()