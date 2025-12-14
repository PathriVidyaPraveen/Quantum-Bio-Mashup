# src/day6_features.py

import numpy as np
import librosa
import pickle
import pywt
from collections import defaultdict
import soundfile as sf
from spectral import extract_normalized_spectrogram


MASTER_DB_PATH = "database/master_db.pkl"
RAW_SONGS_DIR = "raw_songs"
TARGET_SR = 22050

# -----------------------------
# KEY ESTIMATION (SONG-LEVEL)
# -----------------------------
KEYS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def estimate_key(chroma):
    """
    Very simple key estimator:
    Average chroma over time and pick the strongest pitch class.
    """
    chroma_mean = chroma.mean(axis=1)
    key_idx = np.argmax(chroma_mean)
    return KEYS[key_idx]


# -----------------------------
# LOAD SEGMENT DB
# -----------------------------
with open(MASTER_DB_PATH, "rb") as f:
    segments = pickle.load(f)

segments_by_song = defaultdict(list)
for seg in segments:
    segments_by_song[seg.parent_song].append(seg)

print(f"[INFO] Loaded {len(segments)} segments from DB")
print(f"[INFO] Found {len(segments_by_song)} songs")


# -----------------------------
# SONG-LEVEL KEY ASSIGNMENT
# -----------------------------
for song_name, song_segments in segments_by_song.items():
    print(f"[PROCESS] Estimating key for song: {song_name}")

    y = np.load(f"{RAW_SONGS_DIR}/{song_name}.npy")

    chroma = librosa.feature.chroma_cqt(y=y, sr=TARGET_SR)
    song_key = estimate_key(chroma)

    for seg in song_segments:
        seg.key = song_key

    print(f"  → Assigned key: {song_key}")


# -----------------------------
# STFT ATTACHMENT + WAVELETS
# -----------------------------
SCALES = [1, 2, 4, 8]
WAVELET = "morl"

for seg in segments:
    # -------- FIX: ensure spectrogram exists --------
    if seg.spectrogram is None:
        audio, sr = sf.read(seg.wav_path)
        if sr != TARGET_SR:
            raise ValueError(f"SR mismatch in {seg.wav_path}")
        seg.spectrogram = extract_normalized_spectrogram(audio)

    mag = seg.spectrogram              # (1025, 128)
    env = np.mean(mag, axis=0)         # (128,)

    coeffs, freqs = pywt.cwt(
        env,
        scales=SCALES,
        wavelet=WAVELET
    )

    seg.wavelet_energy = np.mean(np.abs(coeffs), axis=1)



# -----------------------------
# SAVE UPDATED DB
# -----------------------------
with open(MASTER_DB_PATH, "wb") as f:
    pickle.dump(segments, f)

print("[DONE] Day 6 features added and DB updated.")



# -----------------------------
# QUICK SANITY CHECK
# -----------------------------
s = segments[0]
print("ID:", s.id)
print("Key:", s.key)
print("Spectrogram shape:", s.spectrogram.shape)
print("Wavelet energy:", s.wavelet_energy)
print("Wavelet shape:", s.wavelet_energy.shape)


