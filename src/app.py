import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dynamic_clouds import DynamicClouds

st.set_page_config(page_title="Nuées dynamiques", layout="wide")
st.title("Implémentation des Nuées dynamiques")
st.caption("Algorithme from scratch — Edwin Diday (1971)")

st.write("### Données de démonstration")
rng = np.random.default_rng(42)
centers = np.array([[0, 0], [5, 5], [0, 6]])
X = np.vstack([c + rng.normal(0, 0.8, size=(80, 2)) for c in centers])
df = pd.DataFrame(X, columns=["X1", "X2"])
st.dataframe(df.head(10), use_container_width=True)

n_features = X.shape[1]

with st.sidebar:
    st.header("Paramètres")
    representation = st.selectbox(
        "Représentation de la nuée",
        ["point", "points", "axes", "distribution", "structure"],
        format_func=lambda x: {
            "point": "Point — cas particulier k-means",
            "points": "Ensemble de points représentatifs",
            "axes": "Axes / composantes factorielles",
            "distribution": "Distribution de probabilités",
            "structure": "Structure représentative",
        }[x],
    )
    k = st.number_input("Nombre de classes", 2, 10, 3)
    p = st.number_input("Nombre de prototypes", 1, 10, 3)
    # Avec n_axes >= nombre de variables, le sous-espace factoriel couvre tout
    # l'espace : le résidu (donc la distance "axes") devient nul pour tout
    # point, quel que soit le cluster -> résultat dégénéré. On borne donc le
    # nombre d'axes à n_features - 1 pour que la réduction de dimension ait
    # un sens sur ces données de démonstration (2 variables ici).
    max_axes = max(1, n_features - 1)
    a = st.number_input("Nombre d'axes", 1, max_axes, min(2, max_axes))
    if representation == "axes" and n_features <= 2:
        st.caption(
            f"Données à {n_features} variables : le nombre d'axes est limité à "
            f"{max_axes} pour garder un résidu non trivial (sinon la distance "
            "'axes' devient nulle pour tous les points)."
        )
    it = st.number_input("Nombre maximal d'itérations", 1, 500, 100)
    run = st.button("Lancer la classification", type="primary")

if run:
    model = DynamicClouds(
        n_clusters=int(k),
        representation=representation,
        n_prototypes=int(p),
        n_axes=int(a),
        max_iter=int(it),
        random_state=42,
    )
    labels = model.fit_predict(X)

    c1, c2, c3 = st.columns(3)
    c1.metric("Itérations", model.n_iter_)
    c2.metric("Inertie", f"{model.inertia_:.3f}")
    c3.metric("Classes", int(k))

    st.write("### Résultats")
    out = df.copy()
    out["Classe"] = labels
    st.dataframe(out, use_container_width=True)

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=labels)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title(f"Nuées dynamiques — {representation}")
    st.pyplot(fig)

    st.write("### Évolution de la fonction objectif")
    fig2, ax2 = plt.subplots()
    ax2.plot(model.history_, marker="o")
    ax2.set_xlabel("Itération")
    ax2.set_ylabel("Inertie / critère")
    ax2.grid(True, alpha=0.25)
    st.pyplot(fig2)

    st.json(model.summary())
