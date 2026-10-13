from pathlib import Path
import logging
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import find_peaks
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

ROOT = Path(__file__).resolve().parents[1]
MAX_CYCLE = 2205
LABELS = {0: "Optimal", 1: "Non optimal"}


def artifact_version(paths):
    # Invalider le cache si les fichiers sont remplacés.
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)


def validate_model(model, feature_names):
    if not isinstance(feature_names, list) or not feature_names or not all(
        isinstance(name, str) for name in feature_names
    ) or len(set(feature_names)) != len(feature_names):
        raise ValueError("La liste des caractéristiques est vide, invalide ou contient des doublons.")
    if not isinstance(model, Pipeline) or len(model.steps) != 2 or not isinstance(model.steps[0][1], StandardScaler):
        raise ValueError("Export attendu : un StandardScaler suivi du classifieur entraîné.")
    scaler, classifier = (step for _, step in model.steps)
    check_is_fitted(scaler)
    check_is_fitted(classifier)
    if not callable(getattr(classifier, "predict_proba", None)):
        raise ValueError("Le modèle exporté ne fournit pas de probabilités.")
    if list(classifier.classes_) != [0, 1]:
        raise ValueError("Le modèle doit utiliser les classes 0 et 1.")
    for step in (scaler, classifier):
        if getattr(step, "n_features_in_", None) != len(feature_names):
            raise ValueError("Le modèle, le scaler et la liste des caractéristiques n'ont pas la même dimension.")
    if not hasattr(scaler, "feature_names_in_") or list(scaler.feature_names_in_) != feature_names:
        raise ValueError("L'ordre des caractéristiques ne correspond pas au scaler. Réexportez les deux fichiers ensemble.")


@st.cache_resource
def load_model(version):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(ROOT / "models" / "final_model.joblib")
            names = joblib.load(ROOT / "models" / "feature_names.joblib")
    except InconsistentVersionWarning as exc:
        raise ValueError(
            f"Modèle exporté avec scikit-learn {exc.original_sklearn_version}, "
            f"mais version installée : {exc.current_sklearn_version}. "
            "Utilisez l'environnement Docker prévu par le projet ou réexportez le modèle avec la version utilisée par l'application."
        ) from exc
    validate_model(model, names)
    return model, names


@st.cache_data
def load_signals(version):
    ps2 = pd.read_csv(ROOT / "data" / "PS2.txt", sep="\t", header=None)
    fs1 = pd.read_csv(ROOT / "data" / "FS1.txt", sep="\t", header=None)
    for name, frame, width in (("PS2", ps2, 6000), ("FS1", fs1, 600)):
        if frame.shape != (MAX_CYCLE, width):
            raise ValueError(f"Dimensions {name} incorrectes : {frame.shape}, attendu : ({MAX_CYCLE}, {width}).")
    return ps2, fs1


def predict_cycle(model, features):
    predicted = model.predict(features)[0]
    probabilities = np.asarray(model.predict_proba(features)[0], dtype=float)
    if predicted not in LABELS or probabilities.shape != (2,) or not np.isfinite(probabilities).all():
        raise ValueError("Le modèle a produit une prédiction invalide.")
    if (probabilities < 0).any() or (probabilities > 1).any() or not np.isclose(probabilities.sum(), 1):
        raise ValueError("Le modèle a produit des probabilités invalides.")
    return int(predicted), dict(zip(model.classes_, probabilities))


def extract_features(df, prefix, sampling_rate, threshold):
    features = pd.DataFrame(index=df.index)

    # Extraire une fenêtre exprimée en secondes
    def window(start, end):
        return df.iloc[
            :,
            int(start * sampling_rate):int(end * sampling_rate)
        ]

    # 1. Statistiques générales
    features[f"{prefix}_mean"] = df.mean(axis=1)
    features[f"{prefix}_median"] = df.median(axis=1)
    features[f"{prefix}_std"] = df.std(axis=1)
    features[f"{prefix}_min"] = df.min(axis=1)
    features[f"{prefix}_max"] = df.max(axis=1)

    features[f"{prefix}_range"] = (
        features[f"{prefix}_max"] - features[f"{prefix}_min"]
    )

    features[f"{prefix}_mean_abs_change"] = (
        df.diff(axis=1).abs().mean(axis=1)
    )

    # 2. Caractéristiques du signal complet
    energy = []
    entropy_values = []
    dominant_frequencies = []
    peak_counts = []
    rolling_mean_std = []
    rolling_std_max = []

    for _, row in df.iterrows():
        signal = row.to_numpy()

        energy.append(np.mean(signal ** 2))

        # Analyse fréquentielle
        centered_signal = signal - signal.mean()
        fft_values = np.fft.rfft(centered_signal)
        frequencies = np.fft.rfftfreq(
            len(signal), d=1 / sampling_rate
        )

        # Exclure la fréquence nulle
        power = np.abs(fft_values[1:]) ** 2
        frequencies = frequencies[1:]

        dominant_frequencies.append(
            frequencies[np.argmax(power)]
        )

        probability = power / (power.sum() + 1e-12)
        entropy = -np.sum(
            probability * np.log2(probability + 1e-12)
        )
        entropy_values.append(
            entropy / np.log2(len(probability))
        )

        # Nombre de pics
        peaks, _ = find_peaks(
            signal,
            prominence=0.5 * np.std(signal),
        )
        peak_counts.append(len(peaks))

        # Fenêtre glissante d'une seconde
        rolling = pd.Series(signal).rolling(
            window=sampling_rate
        )
        rolling_mean_std.append(rolling.mean().std())
        rolling_std_max.append(rolling.std().max())

    features[f"{prefix}_energy"] = energy
    features[f"{prefix}_spectral_entropy"] = entropy_values
    features[f"{prefix}_dominant_frequency"] = dominant_frequencies
    features[f"{prefix}_peak_count"] = peak_counts
    features[f"{prefix}_rolling_mean_std"] = rolling_mean_std
    features[f"{prefix}_rolling_std_max"] = rolling_std_max

    # 3. Caractéristiques autour du premier événement
    first_event = window(8, 12).to_numpy()
    crossings = first_event >= threshold

    # Premier instant où le seuil est atteint
    features[f"{prefix}_crossing_time"] = np.where(
        crossings.any(axis=1),
        8 + crossings.argmax(axis=1) / sampling_rate,
        np.nan,
    )

    # Instant du maximum dans la fenêtre de 8 à 12 secondes
    features[f"{prefix}_first_peak_time"] = (
        8 + first_event.argmax(axis=1) / sampling_rate
    )

    features[f"{prefix}_peak_drop"] = (
        first_event.max(axis=1) - window(11, 12).mean(axis=1)
    )

    # 4. Plateau après le pic
    features[f"{prefix}_plateau_mean"] = window(12, 18).mean(axis=1)
    features[f"{prefix}_plateau_std"] = window(12, 18).std(axis=1)

    # 5. Transitions pendant le cycle
    features[f"{prefix}_dip_depth_20s"] = (
        window(17, 18).mean(axis=1)
        - window(19, 21).min(axis=1)
    )

    features[f"{prefix}_level_change_30s"] = (
        window(31, 33).mean(axis=1)
        - window(27, 28).mean(axis=1)
    )

    features[f"{prefix}_level_change_39s"] = (
        window(40, 42).mean(axis=1)
        - window(37, 38).mean(axis=1)
    )

    # 6. Deux caractéristiques supplémentaires pour PS2
    if prefix == "PS2":
        features["PS2_crossing_to_peak_duration"] = (
            features["PS2_first_peak_time"]
            - features["PS2_crossing_time"]
        )

        features["PS2_drop_49s"] = (
            window(47, 48).mean(axis=1)
            - window(51, 53).mean(axis=1)
        )

    return features

def cycle_features(signal, prefix, sampling_rate):
    threshold = 100 if prefix == "PS2" else 5
    frame = pd.DataFrame([signal])
    return extract_features(frame, prefix, sampling_rate, threshold).iloc[0].to_dict()


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

    frame = pd.DataFrame([features])
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise ValueError(f"Caractéristiques manquantes : {missing}.")

    frame = frame.loc[:, feature_names]

    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Les features calculées contiennent des valeurs invalides.")

    return frame

def main():
    st.set_page_config(page_title="Diagnostic de la valve hydraulique", page_icon="⚙️", layout="wide")
    st.title("Diagnostic de la valve hydraulique")
    st.caption("Projet Machine Learning · PS2 : pression · FS1 : débit")
    st.write("Sélectionnez un cycle enregistré pour estimer l'état de la valve et examiner les mesures.")

    paths = [ROOT / "models" / "final_model.joblib", ROOT / "models" / "feature_names.joblib",
             ROOT / "data" / "PS2.txt", ROOT / "data" / "FS1.txt"]
    try:
        version = artifact_version(paths)
        model, feature_names = load_model(version[:2])
        ps2, fs1 = load_signals(version[2:])
    except FileNotFoundError as exc:
        st.error(f"Fichier nécessaire introuvable : {Path(exc.filename).name if exc.filename else 'fichier du projet'}.")
        st.stop()
    except Exception as exc:
        logging.exception("Chargement de l'application impossible")
        st.error(f"Impossible de charger le projet : {exc}")
        st.stop()

    estimator = model.steps[-1][1]
    names = {"LogisticRegression": "Régression logistique", "SVC": "SVM",
             "RandomForestClassifier": "Forêt aléatoire", "DecisionTreeClassifier": "Arbre de décision"}
    model_name = names.get(type(estimator).__name__, type(estimator).__name__)
    with st.sidebar:
        st.header("Le projet")
        st.write("Un cycle dure 60 secondes.")
        st.markdown("**PS2** : pression en bar, 100 Hz.\n\n**FS1** : débit en L/min, 10 Hz.")
        st.divider()
        st.write("0 : optimal · 1 : non optimal")
        st.caption("La classe non optimale regroupe les conditions 90 %, 80 % et 73 % du jeu de données.")

    for col, label, value in zip(st.columns(3), ["Cycles disponibles", "Modèle chargé", "Caractéristiques"],
                                 [len(ps2), model_name, len(feature_names)]):
        col.metric(label, value)
    diagnostic, details, about = st.tabs(["Diagnostic", "Modèle et caractéristiques", "À propos"])
    with diagnostic:
        with st.form("prediction"):
            cycle = st.number_input("Numéro du cycle", min_value=1, max_value=len(ps2), value=1, step=1)
            submitted = st.form_submit_button("Analyser le cycle", type="primary")
        if submitted:
            st.session_state.pop("diagnostic_result", None)
            try:
                features = prepare_cycle(cycle, ps2, fs1, feature_names)
                prediction, probabilities = predict_cycle(model, features)
                st.session_state["diagnostic_result"] = {
                    "cycle": int(cycle), "prediction": prediction, "probabilities": probabilities,
                    "features": features, "version": version,
                }
            except Exception as exc:
                logging.exception("Prédiction impossible")
                st.error(f"Impossible d'analyser ce cycle : {exc}")
        result = st.session_state.get("diagnostic_result")
        if result is not None and result["version"] != version:
            st.session_state.pop("diagnostic_result", None)
            result = None
            st.info("Les fichiers ont été mis à jour. Relancez l'analyse.")
        if result is None:
            st.info("Choisissez un cycle et cliquez sur Analyser le cycle.")
        else:
            st.subheader(f"Résultat du cycle {result['cycle']}")
            display = st.success if result["prediction"] == 0 else st.warning
            display(f"État estimé : {LABELS[result['prediction']]}")
            cols = st.columns(2)
            for col, label in zip(cols, [0, 1]):
                col.metric(f"Probabilité — {LABELS[label]}", f"{result['probabilities'][label]:.2%}")
            st.caption("Ces probabilités sont celles du modèle ; elles ne représentent pas son taux de bonnes prédictions.")
            pressure = ps2.iloc[result["cycle"] - 1].to_numpy(dtype=float)
            flow = fs1.iloc[result["cycle"] - 1].to_numpy(dtype=float)
            st.subheader("Mesures du cycle")
            zoom = st.checkbox("Observer la première montée : 8 à 12 secondes")
            for col, signal, rate, label, color in zip(st.columns(2), [pressure, flow], [100, 10],
                    ["Pression (bar)", "Débit (L/min)"], ["#168A99", "#c77725"]):
                times = np.arange(len(signal)) / rate
                frame = pd.DataFrame({"Temps (s)": times, label: signal}).set_index("Temps (s)")
                if zoom:
                    frame = frame.loc[(frame.index >= 8) & (frame.index < 12)]
                with col:
                    st.markdown(f"**{label}**")
                    st.line_chart(frame, color=color)
            st.subheader("Franchissements observés")
            for col, name, label in zip(st.columns(2), ["PS2_crossing_time", "FS1_crossing_time"],
                    ["Pression : 100 bar", "Débit : 5 L/min"]):
                if name in result["features"]:
                    col.metric(label, f"{result['features'].iloc[0][name]:.2f} s")
            st.caption("Instants mesurés depuis le début du cycle, dans la fenêtre de 8 à 12 secondes.")
            with st.expander("Caractéristiques transmises au modèle"):
                st.dataframe(result["features"].T.rename(columns={0: "Valeur"}))
                st.download_button("Télécharger les caractéristiques", result["features"].to_csv(index=False),
                                   file_name=f"cycle_{result['cycle']}_features.csv", mime="text/csv")

    with details:
        st.subheader(model_name)
        st.write("Le fichier sauvegardé contient le scaler et le classifieur entraînés. L'application conserve l'ordre des caractéristiques attendu par ce scaler.")
        st.dataframe(pd.DataFrame({"Caractéristique": feature_names}))
        with st.expander("Paramètres du modèle chargé"):
            st.json(estimator.get_params(deep=False))
        st.info("Les scores d'évaluation ne sont pas enregistrés dans ce fichier. Consultez le notebook pour les résultats.")
    with about:
        st.write("L'application classe un cycle enregistré à partir des signaux PS2 et FS1. La cible d'entraînement vaut 0 pour une condition de valve de 100 %, et 1 pour les autres conditions.")
        st.write("Les étiquettes du fichier Profile ne sont pas utilisées pour calculer les prédictions affichées.")
        st.caption("Cette démonstration décrit l'état estimé d'un cycle observé ; elle ne mesure pas un délai avant une panne future.")


if __name__ == "__main__":
    main()
