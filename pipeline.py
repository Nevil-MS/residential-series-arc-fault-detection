import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from scipy.stats import kurtosis, skew


MODEL_PATH = Path(__file__).parent / "data" / "arc_fault_model.joblib"


def extract_features(signal):
    signal = np.asarray(signal).flatten()

    # Time-domain features
    mean = np.mean(signal)
    variance = np.var(signal)
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(signal)
    peak_to_peak = np.ptp(signal)

    crest_factor = peak / rms
    signal_kurtosis = kurtosis(signal)
    signal_skewness = skew(signal)

    # Frequency-domain features
    magnitude = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(len(signal), d=1 / 100_000)

    fft_energy = np.sum(magnitude ** 2)
    high_frequency_energy = np.sum(magnitude[frequencies >= 5000] ** 2)

    # Ignore DC component for these two features
    magnitude[0] = 0

    dominant_frequency = frequencies[np.argmax(magnitude)]
    spectral_centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)

    return pd.DataFrame([{
        "Mean": mean,
        "variance": variance,
        "RMS": rms,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "Crest_Factor": crest_factor,
        "Kurtosis": signal_kurtosis,
        "Skewness": signal_skewness,
        "FFT_Energy": fft_energy,
        "Dominant_Frequency": dominant_frequency,
        "Spectral_Centroid": spectral_centroid,
        "High_Frequency_Energy": high_frequency_energy,
    }])


def predict_signal(signal):
    bundle = joblib.load(MODEL_PATH)
    features = extract_features(signal)[bundle["features"]]

    arc_index = list(bundle["model"].classes_).index(1)
    arc_probability = bundle["model"].predict_proba(features)[0, arc_index]

    return "Arc" if arc_probability >= bundle["threshold"] else "Normal"