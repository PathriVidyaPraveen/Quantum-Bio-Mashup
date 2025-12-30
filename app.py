import streamlit as st
import numpy as np
import pickle
import soundfile as sf
import tempfile
import os
import matplotlib.pyplot as plt

from scipy.linalg import expm


# ===============================
# CONFIG
# ===============================
DB_PATH   = "database/master_db.pkl"
ADJ_PATH  = "database/adjacency_sym.npy"

SR = 22050
DT = 0.05

st.set_page_config(layout="wide")
st.title("Quantum - Biological Mashup Generator")

# ===============================
# LOAD DATABASE (ONCE)
# ===============================
@st.cache_resource
def load_database():
    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)
    db = sorted(db, key=lambda s: s.global_index)
    return db

@st.cache_resource
def load_adjacency():
    A = np.load(ADJ_PATH)
    return A

db = load_database()
A  = load_adjacency()
N  = len(db)

# ===============================
# SIDEBAR CONTROLS
# ===============================
st.sidebar.header("Quantum Controls")

start_idx = st.sidebar.selectbox(
    "Initial Segment",
    range(N),
    format_func=lambda i: db[i].id
)

lambda_noise = st.sidebar.slider(
    "Decoherence λ",
    0.0, 1.0, 0.15
)

lambda_bio = st.sidebar.slider(
    "Bio modulation λ",
    0.0, 1.0, 0.3
)

T_steps = st.sidebar.slider(
    "Quantum Walk Length",
    50, 300, 150
)

generate = st.sidebar.button("Generate Mashup")

# ===============================
# BIO OPERATOR
# ===============================
def build_bio_operator(N):
    # simple toy vibrational spectrum
    v = np.linspace(-1.0, 1.0, N)
    return np.diag(v)

# ===============================
# CTQW + ENAQT
# ===============================
def run_quantum_walk(H, start_idx, T, dt, lambda_noise):
    N = H.shape[0]
    psi = np.zeros(N, dtype=complex)
    psi[start_idx] = 1.0

    probs = np.zeros((T, N))

    deg = np.sum(np.abs(H), axis=1)
    eta = deg / (deg.sum() + 1e-12)

    for t in range(T):
        probs[t] = np.abs(psi)**2

        psi = psi - 1j * (H @ psi) * dt

        if lambda_noise > 0:
            psi = (1 - lambda_noise) * psi + lambda_noise * eta

        psi /= np.linalg.norm(psi)

    return probs

# ===============================
# PATH EXTRACTION
# ===============================
def extract_path(prob, L=20, memory=3):
    path = []
    recent = []

    for t in range(prob.shape[0]):
        p = prob[t].copy()

        for r in recent:
            p[r] = -1

        idx = int(np.argmax(p))
        path.append(idx)
        recent.append(idx)

        if len(recent) > memory:
            recent.pop(0)

        if len(path) >= L:
            break

    return path

# ===============================
# AUDIO RECONSTRUCTION
# ===============================
def crossfade(a, b, cf):
    fade_out = np.linspace(1, 0, cf)
    fade_in  = np.linspace(0, 1, cf)
    a[-cf:] *= fade_out
    b[:cf]  *= fade_in
    return np.concatenate([a[:-cf], a[-cf:] + b[:cf], b[cf:]])

def build_audio(path):
    audio = None
    cf = int(0.08 * SR)

    for idx in path:
        seg = db[idx]
        y, _ = sf.read(seg.wav_path)
        if audio is None:
            audio = y
        else:
            audio = crossfade(audio, y, cf)

    audio /= np.max(np.abs(audio)) + 1e-9
    return audio

# ===============================
# RUN PIPELINE
# ===============================
if generate:
    st.subheader("Running Quantum Evolution")

    D = np.diag(A.sum(axis=1))
    H_base = D - A

    V_bio = build_bio_operator(N)
    H = H_base + lambda_bio * V_bio

    prob = run_quantum_walk(
        H, start_idx,
        T_steps, DT,
        lambda_noise
    )

    path = extract_path(prob)

    audio = build_audio(path)

    # ===============================
    # VISUALS
    # ===============================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Probability Flow")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.imshow(prob.T, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("Time")
        ax.set_ylabel("Segment Index")
        st.pyplot(fig)

    with col2:
        st.subheader("Quantum Mashup")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, audio, SR)
        st.audio(tmp.name)

    st.subheader("Selected Path")
    st.write([db[i].id for i in path])
