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
    st.set_page_config(
        page_title="Projet 1 : Maintenance Predictive",
        layout="wide",
    )
    st.markdown("""
    <style>
    .hero {background: linear-gradient(120deg, #10263d, #125a69);
           padding: 2rem; border-radius: 18px; color: white; margin-bottom: 1.5rem;}
    .hero h1 {color: white; margin: 0; font-size: 2.5rem;}
    .hero p {color: #d4e7ec; max-width: 760px; margin-bottom: 0;}
    .eyebrow {color: #80ddd0; letter-spacing: .12em; font-size: .8rem;
              font-weight: 700; margin-bottom: .6rem;}
    </style>
    <div class="hero">
      <div class="eyebrow">DIAGNOSTIC DU SYSTÈME HYDRAULIQUE</div>
      <h1>MAintenance Predictive</h1>
      <p>Explorez les signaux de pression et de débit pour estimer l’état
      de la valve, cycle par cycle.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("Maintenance Predictive")
        st.caption("Projet Machine Learning · Classification binaire")
        st.divider()
        st.subheader("Les capteurs")
        st.markdown("**PS2 · Pression**\n\n100 Hz · 6 000 mesures par cycle · bar")
        st.markdown("**FS1 · Débit volumique**\n\n10 Hz · 600 mesures par cycle · L/min")
        st.divider()
        st.markdown("**Classes du modèle**\n\n0 → Optimal\n\n1 → Non-optimal")
        st.caption("Un cycle dure 60 secondes. L’analyse utilise les données enregistrées du projet.")

    overview = st.columns(4)
    for col, label, value in zip(
        overview,
        ["Cycles disponibles", "Capteurs", "Durée d’un cycle", "Modèle de référence"],
        [f"{MAX_CYCLE:,}".replace(",", " "), "2", "60 s", "SVM"],
    ):
        col.metric(label, value)

    diagnostic_tab, model_tab, project_tab = st.tabs(
        ["Diagnostic du cycle", "Modèle et caractéristiques", "À propos du projet"]
    )

    with project_tab:
        st.subheader("Comprendre l’état de la valve")
        st.write(
            "Ce projet étudie un système hydraulique à partir de deux séries temporelles : "
            "la pression PS2 et le débit FS1. Le modèle classe chaque cycle en "
            "fonctionnement optimal ou non-optimal."
        )
        left, right = st.columns(2)
        with left:
            st.subheader("Des signaux à la prédiction")
            st.markdown(
                "1. Sélection d’un cycle enregistré.\n"
                "2. Extraction des statistiques et des caractéristiques du signal.\n"
                "3. Standardisation avec le pipeline sauvegardé.\n"
                "4. Classification et estimation des probabilités."
            )
        with right:
            st.subheader("Interprétation des classes")
            st.write("Optimal (0) correspond à une condition de valve égale à 100 dans les données d’entraînement.")
            st.write("Non-optimal (1) regroupe les autres conditions. Cette classe ne précise pas le niveau de dégradation.")
        st.info("Les résultats affichés sont des prédictions du modèle, pas les étiquettes réelles des cycles.")

    with diagnostic_tab:
        st.subheader("Analyser un cycle")
        st.write("Sélectionnez un cycle pour consulter le diagnostic et les mesures qui l’accompagnent.")
        with st.form("prediction"):
            cycle = st.number_input(
                "Numéro de cycle", min_value=1, max_value=MAX_CYCLE, value=1, step=1
            )
            submitted = st.form_submit_button("Analyser le cycle", type="primary")
        if submitted:
            try:
                with st.spinner("Chargement des signaux et analyse du cycle…"):
                    model, feature_names = load_model()
                    ps2, fs1 = load_signals()
                    features = prepare_cycle(cycle, ps2, fs1, feature_names)
                    prediction = int(model.predict(features)[0])
                    by_class = dict(zip(model.classes_, model.predict_proba(features)[0]))
                    # Save only this cycle so the result survives Streamlit reruns.
                    st.session_state["diagnostic_result"] = {
                        "cycle": int(cycle), "prediction": prediction,
                        "probabilities": by_class, "features": features,
                        "pressure": ps2.iloc[cycle - 1].to_numpy(dtype=float),
                        "flow": fs1.iloc[cycle - 1].to_numpy(dtype=float),
                    }
            except FileNotFoundError as exc:
                st.session_state.pop("diagnostic_result", None)
                st.error(f"Fichier introuvable : {exc.filename}")
            except Exception as exc:
                st.session_state.pop("diagnostic_result", None)
                st.error(f"Impossible de réaliser la prédiction : {exc}")

        result = st.session_state.get("diagnostic_result")
        if result is None:
            st.info("Lancez une analyse pour afficher l’état estimé de la valve et les courbes des capteurs.")
        else:
            prediction = result["prediction"]
            probabilities = result["probabilities"]
            label = {0: "Optimal", 1: "Non-optimal"}[prediction]
            st.divider()
            st.subheader(f"Résultat · Cycle {result['cycle']}")
            if prediction == 0:
                st.success("État estimé : Optimal — le modèle classe ce cycle dans le fonctionnement optimal.")
            else:
                st.warning("État estimé : Non-optimal — le modèle classe ce cycle dans le fonctionnement non-optimal.")
            cols = st.columns(3)
            cols[0].metric("Classe prédite", label)
            cols[1].metric("Probabilité Optimal", f"{probabilities[0]:.2%}")
            cols[2].metric("Probabilité Non-optimal", f"{probabilities[1]:.2%}")
            st.caption("Les probabilités sont des estimations du modèle ; elles ne mesurent pas son exactitude sur ce cycle.")

            pressure, flow = result["pressure"], result["flow"]
            st.subheader("Les signaux du cycle")
            left, right = st.columns(2)
            with left:
                st.markdown("**PS2 · Pression**")
                frame = pd.DataFrame({"Temps (s)": np.arange(len(pressure)) / 100, "Pression (bar)": pressure})
                st.line_chart(frame.set_index("Temps (s)"), color="#1a9caa")
                st.caption("6 000 mesures · 100 Hz")
            with right:
                st.markdown("**FS1 · Débit volumique**")
                frame = pd.DataFrame({"Temps (s)": np.arange(len(flow)) / 10, "Débit (L/min)": flow})
                st.line_chart(frame.set_index("Temps (s)"), color="#d48a38")
                st.caption("600 mesures · 10 Hz")

            st.subheader("Indicateurs du cycle")
            cols = st.columns(4)
            cols[0].metric("Pression moyenne", f"{pressure.mean():.2f} bar")
            cols[1].metric("Pression maximale", f"{pressure.max():.2f} bar")
            cols[2].metric("Débit moyen", f"{flow.mean():.2f} L/min")
            cols[3].metric("Débit maximal", f"{flow.max():.2f} L/min")
            with st.expander("Consulter les caractéristiques transmises au modèle"):
                st.dataframe(result["features"].T.rename(columns={0: "Valeur"}), use_container_width=True)

    with model_tab:
        st.subheader("Le pipeline utilisé pour la prédiction")
        st.write("Les caractéristiques sont standardisées par StandardScaler avant d’être transmises au classifieur sauvegardé.")
        st.markdown("**Signaux PS2 + FS1 → Caractéristiques → StandardScaler → Classifieur → État de la valve**")
        try:
            model, feature_names = load_model()
            estimator = model.steps[-1][1]
            params = estimator.get_params()
            cols = st.columns(3)
            cols[0].metric("Classifieur chargé", type(estimator).__name__)
            cols[1].metric("Noyau", str(params.get("kernel", "Sans objet")).upper())
            cols[2].metric("Caractéristiques utilisées", len(feature_names))
            st.caption("Ces informations sont lues directement dans le modèle sauvegardé.")
            with st.expander("Paramètres du classifieur"):
                st.json(params)
            st.subheader("Caractéristiques utilisées")
            st.write("Elles résument le niveau, la dispersion et la dynamique des signaux : moyenne, écart-type, énergie, entropie spectrale, pics et variations temporelles.")
            st.dataframe(pd.DataFrame({"Caractéristique": feature_names}), use_container_width=True)
            st.info("Les performances de validation ne sont pas enregistrées dans ce modèle. Consultez le notebook pour les résultats d’évaluation.")
        except FileNotFoundError as exc:
            st.error(f"Modèle indisponible : {exc.filename}")
        except Exception as exc:
            st.error(f"Impossible de lire le modèle : {exc}")

    st.divider()
    st.caption("Maintenance Predictive · Projet ML · Diagnostic à partir de cycles enregistrés")


if __name__ == "__main__":
    main()
