import argparse
import numpy as np
from dynamic_clouds import DynamicClouds

def make_demo(seed=42):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [5, 5], [0, 6]])
    X = np.vstack([c + rng.normal(0, 0.8, size=(80, 2)) for c in centers])
    return X

def main():
    p = argparse.ArgumentParser(description="Nuées dynamiques from scratch")
    p.add_argument("--representation", choices=["point", "points", "axes", "distribution", "structure"], default="points")
    p.add_argument("--clusters", type=int, default=3)
    p.add_argument("--prototypes", type=int, default=3)
    p.add_argument("--axes", type=int, default=2)
    args = p.parse_args()

    X = make_demo()
    model = DynamicClouds(
        n_clusters=args.clusters,
        representation=args.representation,
        n_prototypes=args.prototypes,
        n_axes=args.axes,
    )
    model.fit(X)
    print(model.summary())
    print("Labels:", model.labels_)

if __name__ == "__main__":
    main()
