# TP — Implémentation des Nuées dynamiques

## Objectif

Implémenter l'algorithme des Nuées dynamiques de Diday (1971) **from scratch**, en montrant que le k-means correspond au cas où chaque classe est représentée par un seul prototype.

## Représentations disponibles

1. `point` : un seul prototype, cas particulier assimilable au k-means.
2. `points` : plusieurs prototypes par classe.
3. `axes` : représentation par un sous-espace factoriel obtenu par ACP calculée avec NumPy.
4. `distribution` : représentation gaussienne par moyenne et matrice de covariance.
5. `structure` : représentation combinant plusieurs prototypes et la dispersion/covariance de la classe.

Le cœur de l'algorithme n'utilise aucune fonction de clustering de scikit-learn.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Exécution console

Depuis la racine :

```bash
python src/main.py --representation points
python src/main.py --representation point
python src/main.py --representation axes
python src/main.py --representation distribution
python src/main.py --representation structure
```

## Interface Streamlit

```bash
streamlit run src/app.py
```

## Tests

```bash
pytest -q
```

## Dépôt GitHub

```bash
git init
git add .
git commit -m "TP Nuées dynamiques from scratch"
git branch -M main
git remote add origin https://github.com/VOTRE_COMPTE/nuees-dynamiques-from-scratch.git
git push -u origin main
```

Remplacer `VOTRE_COMPTE` par le nom du compte GitHub du groupe.
