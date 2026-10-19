import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from pipeline import predict_signal


st.set_page_config(
    page_title="Arc Fault Detection",
    page_icon="⚡",
    layout="wide"
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_PATH = DATA_DIR / "arc_fault_model.joblib"
SAMPLE_RATE = 100_000


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / "test.csv", header=None)

    # Convert CSV cells to numbers. This also handles a CSV
    # whose first row accidentally contains column headings.
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=[df.columns[-1]]).reset_index(drop=True)

    df = df.dropna()
    df.columns = [*range(df.shape[1] - 1), "label"]

    return df


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


df = load_data()
bundle = load_bundle()

normal_samples = df[df["label"] == 0].reset_index(drop=True)
arc_samples = df[df["label"] == 1].reset_index(drop=True)

if normal_samples.empty or arc_samples.empty:
    st.error("The dataset must contain Normal (0) and Arc (1) waveforms.")
    st.stop()


# Session state
if "running" not in st.session_state:
    st.session_state.running = False
if "inject_arc" not in st.session_state:
    st.session_state.inject_arc = False
if "sample_index" not in st.session_state:
    st.session_state.sample_index = 0
if "arc_index" not in st.session_state:
    st.session_state.arc_index = 0
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# Header
st.title("⚡ Arc Fault Detection System")
st.caption("Machine-learning waveform monitoring | Dataset simulation")

start_col, inject_col, stop_col = st.columns(3)

with start_col:
    if st.button("▶ Start Monitoring", use_container_width=True):
        st.session_state.running = True
        st.session_state.inject_arc = False
        st.session_state.last_result = None

with inject_col:
    if st.button("⚡ Inject Arc", use_container_width=True):
        st.session_state.running = True
        st.session_state.inject_arc = True
        st.session_state.arc_index = (
            st.session_state.arc_index + 1
        ) % len(arc_samples)

with stop_col:
    if st.button("⏹ Stop", use_container_width=True):
        st.session_state.running = False


@st.fragment(run_every="2s")
def monitor():
    if not st.session_state.running:
        st.info("Monitoring stopped. Press Start Monitoring to begin.")
        return

    if st.session_state.inject_arc:
        sample = arc_samples.iloc[st.session_state.arc_index]
        source_label = "Arc sample"
    else:
        sample = normal_samples.iloc[st.session_state.sample_index]
        source_label = "Normal sample"

        st.session_state.sample_index = (
            st.session_state.sample_index + 1
        ) % len(normal_samples)

    signal = sample.iloc[:-1].to_numpy(dtype=float)

    # Actual model prediction
    result = predict_signal(signal)

    # Also calculate Arc probability for the dashboard
    features = __import__("pipeline").extract_features(signal)
    features = features[bundle["features"]]
    arc_class_index = list(bundle["model"].classes_).index(1)
    probability = bundle["model"].predict_proba(features)[0, arc_class_index]

    st.markdown(f"### Source: {source_label}")

    # Status indicators
    status_col, probability_col, source_col = st.columns(3)

    with status_col:
        if result == "Arc":
            st.error("🚨 ARC FAULT DETECTED")
        else:
            st.success("Normal")

    with probability_col:
        st.metric("Arc probability", f"{probability:.1%}")

    with source_col:
        st.metric("Samples in window", len(signal))

    # Oscilloscope-style waveform
    time_ms = np.arange(len(signal)) / SAMPLE_RATE * 1000

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#101820")
    ax.set_facecolor("#101820")

    waveform_color = "#ff4b4b" if result == "Arc" else "#00e5a8"

    ax.plot(time_ms, signal, color=waveform_color, linewidth=1.0)
    ax.set_xlim(0, time_ms[-1])
    ax.set_xlabel("Time (ms)", color="white")
    ax.set_ylabel("Amplitude", color="white")
    ax.set_title("Electrical waveform", color="white")
    ax.tick_params(colors="white")
    ax.grid(True, color="#43515c", alpha=0.5)

    for spine in ax.spines.values():
        spine.set_color("#43515c")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if result == "Arc":
        st.error("⚠️ The model classified this waveform as an arc fault.")
    elif st.session_state.inject_arc:
        st.warning("The injected arc sample was classified as Normal.")

    # After showing the injected sample, return to normal monitoring
    if st.session_state.inject_arc:
        st.session_state.inject_arc = False
        st.session_state.last_result = result


monitor()

st.caption(
    "Recorded waveform simulation only. "
    "This application is not connected to physical electrical equipment."
)