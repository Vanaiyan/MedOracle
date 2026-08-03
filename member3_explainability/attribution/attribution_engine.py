"""
member3_explainability/attribution/attribution_engine.py
========================================================
Feature-level attribution with method SELECTION by faithfulness.

Methods
-------
* Integrated Gradients  (PRIMARY)   — captum.IntegratedGradients
* Kernel SHAP           (BASELINE)  — captum.KernelShap
* Intrinsic attention   (CORROB.)   — model's own attention gate

Faithfulness
--------------------------------
For a given attribution, remove (set to baseline) the top-k most important
features and measure the DROP in the target-class probability.  A faithful
explanation points at features the model actually relies on, so removing them
should cut the probability sharply.  We report comprehensiveness averaged over
k = 1..K-1; higher = more faithful.




"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients, KernelShap

from member3_explainability.attribution.model_wrapper import ALL_FEATURES, EMOTIONS


class AttributionEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        feature_names: Optional[List[str]] = None,
        baseline: Optional[torch.Tensor] = None,
    ):
        self.model = model.eval()
        self.feature_names = feature_names or ALL_FEATURES
        n = len(self.feature_names)
        # Baseline = neutral reference input (zeros) unless provided.
        self.baseline = baseline if baseline is not None else torch.zeros(1, n)
        self._ig = IntegratedGradients(self.model)
        self._ks = KernelShap(self.model)

    # -- individual methods ------------------------------------------------

    def integrated_gradients(self, x: torch.Tensor, target: int, n_steps: int = 64) -> Dict[str, float]:
        attr = self._ig.attribute(x, baselines=self.baseline, target=target, n_steps=n_steps)
        return self._to_named(attr)

    def kernel_shap(self, x: torch.Tensor, target: int, n_samples: int = 300) -> Dict[str, float]:
        attr = self._ks.attribute(x, baselines=self.baseline, target=target, n_samples=n_samples)
        return self._to_named(attr)

    def attention(self, x: torch.Tensor) -> Dict[str, float]:
        a = self.model.attention(x)
        return self._to_named(a)

    # -- faithfulness ------------------------------------------------------

    def _target_prob(self, x: torch.Tensor, target: int) -> float:
        with torch.no_grad():
            return F.softmax(self.model(x), dim=-1)[0, target].item()

    def comprehensiveness(self, x: torch.Tensor, target: int, attribution: Dict[str, float]) -> float:
        """
        Mean drop in target probability as the top-1..top-(K-1) attributed
        features are progressively replaced by the baseline. Higher = faithful.
        """
        base_p = self._target_prob(x, target)
        order = sorted(range(len(self.feature_names)),
                       key=lambda i: abs(attribution[self.feature_names[i]]), reverse=True)
        drops = []
        for k in range(1, len(order)):
            x_masked = x.clone()
            for i in order[:k]:
                x_masked[0, i] = self.baseline[0, i]
            drops.append(base_p - self._target_prob(x_masked, target))
        return sum(drops) / len(drops) if drops else 0.0

    # -- selection ---------------------------------------------------------

    def explain(self, x: torch.Tensor, target: int) -> dict:
        """
        Run all three methods, score each by faithfulness, and select the best.
        Returns attributions, per-method faithfulness, and the chosen method.
        """
        methods = {
            "integrated_gradients": self.integrated_gradients(x, target),
            "kernel_shap": self.kernel_shap(x, target),
            "attention": self.attention(x),
        }
        faith = {name: self.comprehensiveness(x, target, attr) for name, attr in methods.items()}
        best = max(faith, key=faith.get)
        return {
            "target_emotion": EMOTIONS[target],
            "attributions": {m: {k: round(v, 4) for k, v in a.items()} for m, a in methods.items()},
            "faithfulness_comprehensiveness": {k: round(v, 4) for k, v in faith.items()},
            "selected_method": best,
            "selected_attribution": {k: round(v, 4) for k, v in methods[best].items()},
        }

    # -- helpers -----------------------------------------------------------

    def _to_named(self, attr: torch.Tensor) -> Dict[str, float]:
        vals = attr.squeeze(0).detach().tolist()
        return {name: float(v) for name, v in zip(self.feature_names, vals)}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from member3_explainability.attribution.model_wrapper import DummyEmotionModel

    torch.manual_seed(3)
    model = DummyEmotionModel(seed=0)
    engine = AttributionEngine(model)

    # ---- single, illustrative case: high eeg_beta + gsr_scr -> 'stress' ---
    x = torch.tensor([[0.2, 0.9, 0.2, 0.9, 0.3, 0.2, 0.1, 0.15, 0.1, 0.2]])
    target = int(torch.argmax(F.softmax(model(x), dim=-1), dim=-1).item())
    out = engine.explain(x, target)
    ig = out["attributions"]["integrated_gradients"]
    top_ig = sorted(ig, key=lambda k: abs(ig[k]), reverse=True)[:3]
    print("=" * 70)
    print(f"DEMO (captum): predicted emotion = '{out['target_emotion']}'")
    print("=" * 70)
    print(f"Top-3 features by Integrated Gradients: {top_ig}")
    print("(planted drivers of 'stress' = eeg_beta, gsr_scr)")

    # ---- faithfulness comparison averaged over many inputs ---------------
    torch.manual_seed(123)
    agg = {"integrated_gradients": [], "kernel_shap": [], "attention": []}
    wins = {k: 0 for k in agg}
    N = 40
    for _ in range(N):
        xi = torch.rand(1, 10)
        ti = int(torch.argmax(F.softmax(model(xi), dim=-1), dim=-1).item())
        r = engine.explain(xi, ti)
        for m, v in r["faithfulness_comprehensiveness"].items():
            agg[m].append(v)
        wins[r["selected_method"]] += 1
    print(f"\nMean faithfulness over {N} inputs (higher = more faithful):")
    for m in agg:
        mean = sum(agg[m]) / len(agg[m])
        print(f"  {m:22s}: {mean:+.4f}   selected {wins[m]}/{N} times")
    best = max(agg, key=lambda m: sum(agg[m]) / len(agg[m]))
    print(f"\n==> Data-backed primary method: {best}")
