"""
Implémentation from scratch des Nuées dynamiques.

Aucune bibliothèque de clustering n'est utilisée.
Dépendances autorisées : numpy uniquement pour les calculs numériques.

Représentations :
- point       : 1 prototype => cas particulier des k-means
- points      : plusieurs prototypes par classe
- axes        : sous-espace factoriel obtenu par ACP from scratch
- distribution: gaussienne (moyenne + covariance réguliarisée)
- structure   : prototypes + matrice de dispersion, représentant la forme du nuage
"""

from __future__ import annotations
import numpy as np


class DynamicClouds:
    def __init__(
        self,
        n_clusters=3,
        representation="points",
        n_prototypes=3,
        n_axes=2,
        max_iter=100,
        tol=1e-4,
        random_state=42,
        reg=1e-6,
    ):
        self.n_clusters = int(n_clusters)
        self.representation = representation
        self.n_prototypes = int(n_prototypes)
        self.n_axes = int(n_axes)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.reg = float(reg)

        self.representations_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.history_ = []

    # ---------- Initialisation ----------
    def _init_prototypes(self, X, rng):
        n = len(X)
        ids = [rng.integers(n)]
        for _ in range(1, self.n_prototypes):
            D = np.min(
                np.sum((X[:, None, :] - X[np.array(ids)][None, :, :]) ** 2, axis=2),
                axis=1,
            )
            probs = D / (D.sum() + 1e-12)
            ids.append(rng.choice(n, p=probs))
        return X[np.array(ids)].copy()

    def _init_representations(self, X, rng):
        """Construit une représentation initiale déjà conforme au type choisi.

        Bug corrigé : cette méthode ne produisait jusqu'ici que des dicts
        {"prototypes": ...}, quelle que soit la représentation demandée. Or
        le tout premier appel de fit() effectue un _assign() (donc un
        _distance()) AVANT le premier _update() : pour "axes", "distribution"
        et "structure", _distance() cherche des clés ("mean", "axes", "cov")
        qui n'existaient pas encore -> KeyError dès la première itération.
        On construit donc ici, pour chaque cluster, un petit échantillon
        initial de points (via l'init façon k-means++ existante) puis on en
        dérive la représentation adéquate avec les mêmes outils que
        _update() (_pca / _safe_cov), pour rester cohérent avec le reste de
        l'algorithme.
        """
        reps = []
        for _ in range(self.n_clusters):
            P = self._init_prototypes(X, rng)

            if self.representation in ("point", "points"):
                reps.append({"prototypes": P})

            elif self.representation == "axes":
                na = min(self.n_axes, X.shape[1])
                mean, axes = self._pca(P, na)
                reps.append({"mean": mean, "axes": axes})

            elif self.representation == "distribution":
                reps.append({"mean": P.mean(axis=0), "cov": self._safe_cov(P)})

            elif self.representation == "structure":
                reps.append({"prototypes": P, "cov": self._safe_cov(P)})

            else:
                raise ValueError(f"Représentation inconnue: {self.representation}")

        return reps

    # ---------- Outils numériques ----------
    def _safe_cov(self, X):
        d = X.shape[1]
        if len(X) <= 1:
            cov = np.eye(d)
        else:
            cov = np.cov(X, rowvar=False)
            cov = np.atleast_2d(cov)
            if cov.shape != (d, d):
                cov = np.eye(d) * float(np.var(X))
        cov = cov + self.reg * np.eye(d)
        return cov

    def _pca(self, X, n_axes):
        mean = X.mean(axis=0)
        Z = X - mean
        if len(X) <= 1:
            axes = np.eye(X.shape[1])[:, :n_axes]
            return mean, axes
        cov = (Z.T @ Z) / max(1, len(X) - 1)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        axes = vecs[:, order[:n_axes]]
        return mean, axes

    # ---------- Distances ----------
    def _distance(self, x, rep):
        if self.representation in ("point", "points"):
            P = rep["prototypes"]
            d2 = np.sum((P - x) ** 2, axis=1)
            return float(np.min(d2))

        if self.representation == "axes":
            mean = rep["mean"]
            A = rep["axes"]
            z = x - mean
            proj = A @ (A.T @ z)
            residual = z - proj
            return float(residual @ residual)

        if self.representation == "distribution":
            mu, cov = rep["mean"], rep["cov"]
            z = x - mu
            try:
                inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(cov)
            return float(z @ inv @ z)

        if self.representation == "structure":
            P = rep["prototypes"]
            cov = rep["cov"]
            z = x - P.mean(axis=0)
            try:
                inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(cov)
            # Terme local + terme global de forme.
            local = np.min(np.sum((P - x) ** 2, axis=1))
            shape = z @ inv @ z
            return float(local + 0.25 * shape)

        raise ValueError(f"Représentation inconnue: {self.representation}")

    # ---------- Mise à jour ----------
    def _update(self, X, labels, old_reps, rng):
        reps = []
        for k in range(self.n_clusters):
            C = X[labels == k]
            if len(C) == 0:
                reps.append(old_reps[k])
                continue

            if self.representation == "point":
                reps.append({"prototypes": C.mean(axis=0, keepdims=True)})

            elif self.representation == "points":
                m = min(self.n_prototypes, len(C))
                # Petit k-means interne from scratch pour obtenir m prototypes.
                P = C[rng.choice(len(C), size=m, replace=False)].copy()
                for _ in range(30):
                    D = ((C[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
                    a = D.argmin(axis=1)
                    newP = P.copy()
                    for j in range(m):
                        if np.any(a == j):
                            newP[j] = C[a == j].mean(axis=0)
                    if np.max(np.abs(newP - P)) < self.tol:
                        P = newP
                        break
                    P = newP
                reps.append({"prototypes": P})

            elif self.representation == "axes":
                na = min(self.n_axes, C.shape[1])
                mean, axes = self._pca(C, na)
                reps.append({"mean": mean, "axes": axes})

            elif self.representation == "distribution":
                reps.append({"mean": C.mean(axis=0), "cov": self._safe_cov(C)})

            elif self.representation == "structure":
                m = min(self.n_prototypes, len(C))
                P = C[rng.choice(len(C), size=m, replace=False)].copy()
                for _ in range(30):
                    D = ((C[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
                    a = D.argmin(axis=1)
                    newP = P.copy()
                    for j in range(m):
                        if np.any(a == j):
                            newP[j] = C[a == j].mean(axis=0)
                    if np.max(np.abs(newP - P)) < self.tol:
                        P = newP
                        break
                    P = newP
                reps.append({"prototypes": P, "cov": self._safe_cov(C)})

        return reps

    def _assign(self, X, reps):
        D = np.array([[self._distance(x, r) for r in reps] for x in X])
        return D.argmin(axis=1), D

    # ---------- Algorithme principal ----------
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X doit être une matrice 2D non vide.")
        if self.n_clusters < 2 or self.n_clusters > len(X):
            raise ValueError("n_clusters doit être compris entre 2 et le nombre d'individus.")

        rng = np.random.default_rng(self.random_state)
        reps = self._init_representations(X, rng)
        self.history_ = []

        for it in range(1, self.max_iter + 1):
            labels, D = self._assign(X, reps)
            new_reps = self._update(X, labels, reps, rng)
            inertia = float(np.sum(np.min(D, axis=1)))
            self.history_.append(inertia)

            # Critère de stabilité : labels inchangés ou variation faible de l'objectif.
            if self.labels_ is not None and np.array_equal(labels, self.labels_):
                reps = new_reps
                self.n_iter_ = it
                break

            if len(self.history_) > 1:
                delta = abs(self.history_[-2] - self.history_[-1])
                if delta < self.tol:
                    reps = new_reps
                    self.n_iter_ = it
                    break

            self.labels_ = labels.copy()
            reps = new_reps
            self.n_iter_ = it

        self.labels_, D = self._assign(X, reps)
        self.representations_ = reps
        self.inertia_ = float(np.sum(np.min(D, axis=1)))
        return self

    def predict(self, X):
        if self.representations_ is None:
            raise RuntimeError("Le modèle doit être entraîné avec fit() avant predict().")
        X = np.asarray(X, dtype=float)
        labels, _ = self._assign(X, self.representations_)
        return labels

    def fit_predict(self, X):
        return self.fit(X).labels_

    def summary(self):
        if self.representations_ is None:
            return {}
        return {
            "representation": self.representation,
            "n_clusters": self.n_clusters,
            "n_iter": self.n_iter_,
            "inertia": self.inertia_,
            "cluster_sizes": np.bincount(self.labels_, minlength=self.n_clusters).tolist(),
        }
