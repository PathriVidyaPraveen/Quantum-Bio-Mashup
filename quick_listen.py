# quick_listen.py
import soundfile as sf
import numpy as np

files = [
    "database/audio_segments/retro-lounge-389644_bar_07.wav",
    "database/audio_segments/retro-lounge-389644_bar_06.wav",
    "database/audio_segments/retro-lounge-389644_bar_04.wav",
    "database/audio_segments/retro-lounge-389644_bar_03.wav",
]

audio = []
sr_ref = None

for f in files:
    y, sr = sf.read(f)
    if sr_ref is None:
        sr_ref = sr
    audio.append(y)

mix = np.concatenate(audio)
sf.write("outputs/debug_concat.wav", mix, sr_ref)
