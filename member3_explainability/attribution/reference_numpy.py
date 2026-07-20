"""
member3_explainability/attribution/reference_numpy.py
NumPy-only REFERENCE for the Phase 2 attribution engine.

attribution_engine.py is the production path (captum IntegratedGradients +
KernelShap over the real PyTorch models). This reproduces the SAME methodology
with zero heavy deps so it can be run/validated before torch is installed:
  * Integrated Gradients  (Riemann sum of finite-difference gradients)
  * Exact Shapley         (all 2^F coalitions; F=10 -> 1024; exact, not sampled)
  * Intrinsic attention   (model's own softmax feature gate)
  * Faithfulness          (comprehensiveness: prob drop when top-k removed)
The winning method is selected by measured faithfulness -- the data-backed
answer to "why not just SHAP?".

Author : Adshaya Balarajah (214024V)
"""
from __future__ import annotations
from itertools import combinations
from math import factorial
from typing import Dict, List
import numpy as np

PHYSIO_FEATURES = ["eeg_alpha", "eeg_beta", "eeg_theta", "gsr_scr", "gsr_tonic"]
VIDEO_FEATURES = ["au_smile", "au_brow", "au_eye", "head_pose", "face_valence"]
ALL_FEATURES = PHYSIO_FEATURES + VIDEO_FEATURES
EMOTIONS = ["stress", "calm", "happy", "sad", "angry"]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class NumpyEmotionModel:
    """Model with softmax feature-attention gate and PLANTED feature->class
    structure (so attribution has real reliance to recover). Planted drivers:
    stress<-eeg_beta,gsr_scr ; calm<-eeg_alpha,gsr_tonic ; happy<-au_smile,face_valence ;
    sad<-au_brow,eeg_theta ; angry<-au_brow,gsr_scr. Mild quadratic -> nonlinear."""

    def __init__(self, n_features: int = 10, n_classes: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.Wa = rng.normal(0, 0.6, (n_features, n_features))
        self.ba = rng.normal(0, 0.3, n_features)
        idx = {f: i for i, f in enumerate(ALL_FEATURES)}
        M = np.zeros((n_classes, n_features))
        planted = {0: ["eeg_beta", "gsr_scr"], 1: ["eeg_alpha", "gsr_tonic"],
                   2: ["au_smile", "face_valence"], 3: ["au_brow", "eeg_theta"],
                   4: ["au_brow", "gsr_scr"]}
        for c, feats in planted.items():
            for f in feats:
                M[c, idx[f]] = 2.0
        self.M = M + rng.normal(0, 0.05, M.shape)
        self._last_attn = None

    def logits(self, x: np.ndarray) -> np.ndarray:
        a = _softmax(self.Wa @ x + self.ba)
        self._last_attn = a
        g = x * a
        return self.M @ g + 0.5 * (self.M @ (g ** 2))

    def prob(self, x: np.ndarray, target: int) -> float:
        return float(_softmax(self.logits(x))[target])

    def attention(self, x: np.ndarray) -> np.ndarray:
        self.logits(x)
        return self._last_attn


class ReferenceAttributionEngine:
    def __init__(self, model, feature_names=None, baseline=None):
        self.model = model
        self.feature_names = feature_names or ALL_FEATURES
        self.n = len(self.feature_names)
        self.baseline = np.zeros(self.n) if baseline is None else baseline

    def _grad(self, x, target, eps=1e-4):
        g = np.zeros(self.n)
        for i in range(self.n):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps; xm[i] -= eps
            g[i] = (self.model.prob(xp, target) - self.model.prob(xm, target)) / (2 * eps)
        return g

    def integrated_gradients(self, x, target, steps=64):
        total = np.zeros(self.n)
        for s in range(1, steps + 1):
            alpha = s / steps
            total += self._grad(self.baseline + alpha * (x - self.baseline), target)
        attr = (x - self.baseline) * total / steps
        return dict(zip(self.feature_names, attr))

    def exact_shapley(self, x, target):
        n = self.n; idx = list(range(n))
        def v(mask):
            xs = self.baseline.copy()
            for i in mask:
                xs[i] = x[i]
            return self.model.prob(xs, target)
        phi = np.zeros(n)
        for i in idx:
            others = [j for j in idx if j != i]
            for k in range(len(others) + 1):
                w = factorial(k) * factorial(n - k - 1) / factorial(n)
                for S in combinations(others, k):
                    phi[i] += w * (v(set(S) | {i}) - v(set(S)))
        return dict(zip(self.feature_names, phi))

    def attention(self, x):
        return dict(zip(self.feature_names, self.model.attention(x)))

    def comprehensiveness(self, x, target, attr):
        base_p = self.model.prob(x, target)
        order = sorted(range(self.n), key=lambda i: abs(attr[self.feature_names[i]]), reverse=True)
        drops = []
        for k in range(1, self.n):
            xm = x.copy()
            for i in order[:k]:
                xm[i] = self.baseline[i]
            drops.append(base_p - self.model.prob(xm, target))
        return float(np.mean(drops))

    def explain(self, x, target):
        methods = {
            "integrated_gradients": self.integrated_gradients(x, target),
            "exact_shapley": self.exact_shapley(x, target),
            "attention": self.attention(x),
        }
        faith = {m: self.comprehensiveness(x, target, a) for m, a in methods.items()}
        best = max(faith, key=faith.get)
        return {
            "target_emotion": EMOTIONS[target],
            "attributions": {m: {k: round(v, 4) for k, v in a.items()} for m, a in methods.items()},
            "faithfulness_comprehensiveness": {k: round(v, 4) for k, v in faith.items()},
            "selected_method": best,
        }


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    model = NumpyEmotionModel(seed=0)
    engine = ReferenceAttributionEngine(model, baseline=np.zeros(10))
    x = np.array([0.2, 0.9, 0.2, 0.9, 0.3, 0.2, 0.1, 0.15, 0.1, 0.2])
    target = int(np.argmax(_softmax(model.logits(x))))
    out = engine.explain(x, target)
    ig = out["attributions"]["integrated_gradients"]
    top_ig = sorted(ig, key=lambda k: abs(ig[k]), reverse=True)[:3]
    print("=" * 70)
    print(f"DEMO (NumPy reference): predicted emotion = '{out['target_emotion']}'")
    print("=" * 70)
    print(f"Top-3 features by Integrated Gradients: {top_ig}")
    print("(planted drivers of 'stress' = eeg_beta, gsr_scr)")
    phi = engine.exact_shapley(x, target)
    v_all, v_empty = model.prob(x, target), model.prob(engine.baseline, target)
    print(f"\nExact-Shapley efficiency: sum(phi)={sum(phi.values()):.6f} == v(all)-v(empty)={v_all - v_empty:.6f}")
    agg = {"integrated_gradients": [], "exact_shapley": [], "attention": []}
    wins = {"integrated_gradients": 0, "exact_shapley": 0, "attention": 0}
    N = 60
    for _ in range(N):
        xi = rng.uniform(0, 1, 10)
        ti = int(np.argmax(_softmax(model.logits(xi))))
        r = engine.explain(xi, ti)
        for m, val in r["faithfulness_comprehensiveness"].items():
            agg[m].append(val)
        wins[r["selected_method"]] += 1
    print(f"\nMean faithfulness over {N} inputs (higher = more faithful):")
    for m in agg:
        print(f"  {m:22s}: {np.mean(agg[m]):+.4f}   selected {wins[m]}/{N} times")
    best = max(agg, key=lambda m: np.mean(agg[m]))
    print(f"\n==> Data-backed primary method: {best}")
