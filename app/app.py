from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import find_peaks
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
MAX_CYCLE = 2205

REMOVED_FEATURES = [
    "PS2_min",
    "PS2_dominant_frequency",
    "PS2_range",
    "FS1_min",
    "FS1_range",
]


@st.cache_resource
def load_model():
    model = joblib.load(ROOT / "models" / "svm_model.joblib")
    feature_names = joblib.load(ROOT / "models" / "feature_names.joblib")

    if not isinstance(model, Pipeline) or not any(
        isinstance(step, StandardScaler)
        for _, step in model.steps
    ):
        raise ValueError(
            "Le modèle doit contenir le pipeline StandardScaler + SVM. "
            "Réexportez-le depuis le notebook."
        )

    if not isinstance(feature_names, list) or not all(
        isinstance(name, str) for name in feature_names
    ):
        raise ValueError("La liste des features sauvegardées est invalide.")

    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("La liste des features est vide ou contient des doublons.")

    if list(model.classes_) != [0, 1]:
        raise ValueError("Les classes du modèle doivent être 0 et 1.")

    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != feature_names:
            raise ValueError(
                "Le modèle et feature_names.joblib ne correspondent pas."
            )

    return model, feature_names


@st.cache_data
def load_signals():
    ps2 = pd.read_csv(ROOT / "data" / "PS2.txt", sep="\t", header=None)
    fs1 = pd.read_csv(ROOT / "data" / "FS1.txt", sep="\t", header=None)

    if ps2.shape != (MAX_CYCLE, 6000):
        raise ValueError(
            f"Dimensions PS2 incorrectes : {ps2.shape}, "
            f"attendu : ({MAX_CYCLE}, 6000)."
        )

    if fs1.shape != (MAX_CYCLE, 600):
        raise ValueError(
            f"Dimensions FS1 incorrectes : {fs1.shape}, "
            f"attendu : ({MAX_CYCLE}, 600)."
        )

    return ps2, fs1


def cycle_features(signal, prefix, sampling_rate):
    """Calculs identiques à extract_features du notebook."""
    series = pd.Series(signal)

    centered_signal = signal - np.mean(signal)
    fft_values = np.fft.rfft(centered_signal)
    power = (np.abs(fft_values) ** 2)[1:]
    frequencies = np.fft.rfftfreq(
        len(signal), d=1 / sampling_rate
    )[1:]

    probability = power / (power.sum() + 1e-12)
    entropy = -np.sum(
        probability * np.log2(probability + 1e-12)
    )
    entropy = entropy / np.log2(len(probability))

    peaks, _ = find_peaks(
        signal,
        prominence=0.5 * np.std(signal),
    )

    rolling_mean = series.rolling(window=sampling_rate).mean()
    rolling_std = series.rolling(window=sampling_rate).std()

    values = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
        "range": series.max() - series.min(),
        "mean_abs_change": series.diff().abs().mean(),
        "energy": np.mean(signal ** 2),
        "spectral_entropy": entropy,
        "dominant_frequency": frequencies[np.argmax(power)],
        "peak_count": len(peaks),
        "rolling_mean_std": rolling_mean.std(),
        "rolling_std_max": rolling_std.max(),
    }

    return {
        f"{prefix}_{name}": value
        for name, value in values.items()
    }


def prepare_cycle(cycle, ps2, fs1, feature_names):
    if (
        isinstance(cycle, bool)
        or not isinstance(cycle, (int, np.integer))
        or not 1 <= cycle <= MAX_CYCLE
    ):
        raise ValueError("Le numéro de cycle doit être un entier entre 1 et 2205.")

    index = cycle - 1
    ps2_signal = ps2.iloc[index].to_numpy(dtype=float)
    fs1_signal = fs1.iloc[index].to_numpy(dtype=float)

    if not (
        np.isfinite(ps2_signal).all()
        and np.isfinite(fs1_signal).all()
    ):
        raise ValueError("Ce cycle contient des valeurs manquantes ou infinies.")

    features = {
        **cycle_features(ps2_signal, "PS2", 100),
        **cycle_features(fs1_signal, "FS1", 10),
    }

    frame = pd.DataFrame([features]).drop(columns=REMOVED_FEATURES)

    if set(frame.columns) != set(feature_names):
        missing = sorted(set(feature_names) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(feature_names))
        raise ValueError(
            f"Features incompatibles. Manquantes : {missing}. "
            f"Supplémentaires : {extra}."
        )

    frame = frame.loc[:, feature_names]

    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Les features calculées contiennent des valeurs invalides.")

    return frame


def main():
    st.set_page_config(page_title="État de la valve", page_icon="🔍")
    st.title("Prédiction de l’état de la valve")
    st.write("Sélectionnez un cycle pour analyser les signaux PS2 et FS1.")

    with st.form("prediction"):
        cycle = st.number_input(
            "Numéro de cycle",
            min_value=1,
            max_value=MAX_CYCLE,
            value=1,
            step=1,
        )
        submitted = st.form_submit_button("Prédire")

    if not submitted:
        return

    try:
        with st.spinner("Analyse du cycle…"):
            model, feature_names = load_model()
            ps2, fs1 = load_signals()
            features = prepare_cycle(cycle, ps2, fs1, feature_names)

            prediction = int(model.predict(features)[0])
            probabilities = model.predict_proba(features)[0]
            by_class = dict(zip(model.classes_, probabilities))

        label = {0: "Optimal", 1: "Non-optimal"}[prediction]
        st.subheader(f"Cycle {cycle} : {label}")
        st.metric(
            "Probabilité de la classe prédite",
            f"{by_class[prediction]:.2%}",
        )

        left, right = st.columns(2)
        left.metric("Optimal (0)", f"{by_class[0]:.2%}")
        right.metric("Non-optimal (1)", f"{by_class[1]:.2%}")
        st.caption("Les probabilités sont des estimations du modèle.")

    except FileNotFoundError as exc:
        st.error(f"Fichier introuvable : {exc.filename}")
    except Exception as exc:
        st.error(f"Impossible de réaliser la prédiction : {exc}")


if __name__ == "__main__":
    main()