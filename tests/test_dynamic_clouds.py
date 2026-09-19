import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1] / "src"))

import numpy as np
from dynamic_clouds import DynamicClouds

def data():
    rng = np.random.default_rng(1)
    return np.vstack([
        rng.normal([0, 0], 0.4, (30, 2)),
        rng.normal([4, 4], 0.4, (30, 2)),
        rng.normal([0, 5], 0.4, (30, 2)),
    ])

def test_all_representations():
    X = data()
    for rep in ["point", "points", "axes", "distribution", "structure"]:
        m = DynamicClouds(3, representation=rep, random_state=42)
        y = m.fit_predict(X)
        assert len(y) == len(X)
        assert m.inertia_ >= 0

if __name__ == "__main__":
    test_all_representations()
    print("Tests OK")
