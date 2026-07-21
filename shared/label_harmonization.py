"""
shared/label_harmonization.py
==============================
MedOracle — Label Harmonization Module

Converts raw dataset labels from DEAP and CREMA-D into the shared
five-class emotion taxonomy used by the entire MedOracle pipeline.

Emotion classes (LOCKED — do NOT add a 6th class):
    stress  → 0
    calm    → 1
    happy   → 2
    sad     → 3
    angry   → 4

Usage
-----
    from shared.label_harmonization import map_deap_to_class, map_cremad_to_class

    # DEAP
    label = map_deap_to_class(valence=6.2, arousal=7.1, dominance=3.4)
    # → "stress"

    # CREMA-D
    label = map_cremad_to_class("FEA")
    # → "stress"

    label = map_cremad_to_class("DIS")
    # → None   (DIS is dropped from training)

Academic References
-------------------
- Russell (1980)     — Circumplex model of affect (V/A thresholds)
- Mehrabian (1996)   — PAD emotional state model (dominance axis)
- Kreibig (2010)     — Fear → sympathetic activation (FEA → stress mapping)
- Koelstra et al. (2012) — DEAP dataset
- Cao et al. (2014)  — CREMA-D dataset

Author: MedOracle Team (214206G · 214024V · 214215H)
"""

from typing import Optional

# ---------------------------------------------------------------------------
# Shared constants — import these instead of redefining elsewhere
# ---------------------------------------------------------------------------

EMOTION_CLASSES: dict[str, int] = {
    "stress": 0,
    "calm":   1,
    "happy":  2,
    "sad":    3,
    "angry":  4,
}

_THRESHOLD = 5.0   # midpoint of the 1–9 DEAP rating scale


# ---------------------------------------------------------------------------
# DEAP mapping  (Valence / Arousal / Dominance → 5-class)
# ---------------------------------------------------------------------------

# Decision table based on Russell (1980) circumplex + Mehrabian (1996) PAD model
#
#  Label  | Valence | Arousal | Dominance
#  --------+---------+---------+----------
#  stress  |  < 5   |  ≥ 5   |  < 5
#  calm    |  ≥ 5   |  < 5   |  ≥ 5
#  happy   |  ≥ 5   |  ≥ 5   |  ≥ 5
#  sad     |  < 5   |  < 5   |  < 5
#  angry   |  < 5   |  ≥ 5   |  ≥ 5
#
# Boundary rule: values exactly == 5 are treated as ≥ 5 (positive side).
# Unclassifiable combinations (V≥5, A<5, D<5) → return None (rare edge case).

def map_deap_to_class(
    valence:   float,
    arousal:   float,
    dominance: float,
) -> Optional[str]:
    """
    Map a DEAP (Valence, Arousal, Dominance) triple to an emotion class label.

    Parameters
    ----------
    valence   : float in [1, 9]
    arousal   : float in [1, 9]
    dominance : float in [1, 9]

    Returns
    -------
    str   — one of "stress", "calm", "happy", "sad", "angry"
    None  — if the V/A/D combination does not cleanly map to any class
            (e.g. V≥5, A<5, D<5 — low-V, low-A combos outside the model)

    Raises
    ------
    ValueError if any input is outside [1, 9]

    Examples
    --------
    >>> map_deap_to_class(3.0, 7.0, 2.0)
    'stress'
    >>> map_deap_to_class(7.0, 3.0, 8.0)
    'calm'
    >>> map_deap_to_class(7.0, 7.0, 7.0)
    'happy'
    >>> map_deap_to_class(2.0, 2.0, 2.0)
    'sad'
    >>> map_deap_to_class(2.0, 7.0, 7.0)
    'angry'
    >>> map_deap_to_class(4.5, 4.5, 4.5)   # boundary → positive side
    'happy'
    """
    _validate_deap_range(valence,   "valence")
    _validate_deap_range(arousal,   "arousal")
    _validate_deap_range(dominance, "dominance")

    v_hi = valence   >= _THRESHOLD   # True → high valence
    a_hi = arousal   >= _THRESHOLD   # True → high arousal
    d_hi = dominance >= _THRESHOLD   # True → high dominance

    # --- 5 canonical mappings ---
    if     v_hi and     a_hi and     d_hi:  return "happy"
    if     v_hi and not a_hi and     d_hi:  return "calm"
    if not v_hi and     a_hi and not d_hi:  return "stress"
    if not v_hi and not a_hi and not d_hi:  return "sad"
    if not v_hi and     a_hi and     d_hi:  return "angry"

    # --- Russell (1980) extension for happy ---
    # V_hi, A_hi, D_lo → happy (Russell's Circumplex defines happy as high V + high A,
    # without requiring high dominance. Dominance is Mehrabian's PAD addition.
    # Using Russell's simpler 2D definition here recovers previously unclassified
    # high-valence high-arousal samples, increasing happy class representation.)
    if v_hi and a_hi and not d_hi:  return "happy"

    # --- remaining unclassifiable combination ---
    # V_hi, A_lo, D_lo → no standard emotion label in either model
    return None


def map_deap_to_class_int(
    valence:   float,
    arousal:   float,
    dominance: float,
) -> Optional[int]:
    """
    Same as map_deap_to_class but returns the integer class index
    (from EMOTION_CLASSES) instead of the string label.

    Returns None for unclassifiable combinations.
    """
    label = map_deap_to_class(valence, arousal, dominance)
    return EMOTION_CLASSES[label] if label is not None else None


# ---------------------------------------------------------------------------
# CREMA-D mapping  (label string → 5-class)
# ---------------------------------------------------------------------------

# Mapping table — Kreibig (2010) justifies FEA → stress
# DIS has no clean mapping and is DROPPED from training.

_CREMAD_MAP: dict[str, Optional[str]] = {
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "NEU": "calm",
    "FEA": "stress",
    "DIS": None,        # dropped — no clean mapping
}


def map_cremad_to_class(label_str: str) -> Optional[str]:
    """
    Map a CREMA-D emotion label to a MedOracle class label.

    Parameters
    ----------
    label_str : str
        One of the six CREMA-D labels: "ANG", "DIS", "FEA", "HAP", "NEU", "SAD".
        Case-insensitive.

    Returns
    -------
    str   — one of "stress", "calm", "happy", "sad", "angry"
    None  — for "DIS" (disgust, which is dropped from training)

    Raises
    ------
    ValueError if label_str is not a recognised CREMA-D label.

    Examples
    --------
    >>> map_cremad_to_class("ANG")
    'angry'
    >>> map_cremad_to_class("FEA")
    'stress'
    >>> map_cremad_to_class("NEU")
    'calm'
    >>> map_cremad_to_class("DIS")   # dropped
    None
    >>> map_cremad_to_class("dis")   # case-insensitive
    None
    """
    normalised = label_str.strip().upper()
    if normalised not in _CREMAD_MAP:
        raise ValueError(
            f"Unknown CREMA-D label: '{label_str}'. "
            f"Expected one of {list(_CREMAD_MAP.keys())}."
        )
    return _CREMAD_MAP[normalised]


def map_cremad_to_class_int(label_str: str) -> Optional[int]:
    """
    Same as map_cremad_to_class but returns the integer class index.
    Returns None for "DIS" (dropped label).
    """
    label = map_cremad_to_class(label_str)
    return EMOTION_CLASSES[label] if label is not None else None


# ---------------------------------------------------------------------------
# Batch helpers (useful during dataset loading)
# ---------------------------------------------------------------------------

def batch_map_deap(trials: list[tuple[float, float, float]]) -> list[Optional[str]]:
    """
    Apply map_deap_to_class to a list of (valence, arousal, dominance) tuples.

    Parameters
    ----------
    trials : list of (valence, arousal, dominance) tuples

    Returns
    -------
    list of str|None — same length as input
    """
    return [map_deap_to_class(v, a, d) for v, a, d in trials]


def batch_map_cremad(labels: list[str]) -> list[Optional[str]]:
    """
    Apply map_cremad_to_class to a list of CREMA-D label strings.

    Parameters
    ----------
    labels : list of str

    Returns
    -------
    list of str|None — same length as input; None entries = DIS (dropped)
    """
    return [map_cremad_to_class(lbl) for lbl in labels]


# ---------------------------------------------------------------------------
# Internal validators
# ---------------------------------------------------------------------------

def _validate_deap_range(value: float, name: str) -> None:
    if not (1.0 <= value <= 9.0):
        raise ValueError(
            f"DEAP rating '{name}' must be in [1, 9], got {value}"
        )


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== DEAP mapping ===")
    cases = [
        (3.0, 7.0, 2.0, "stress"),
        (7.0, 3.0, 8.0, "calm"),
        (7.0, 7.0, 7.0, "happy"),
        (2.0, 2.0, 2.0, "sad"),
        (2.0, 7.0, 7.0, "angry"),
        (5.0, 5.0, 5.0, "happy"),   # boundary → happy
    ]
    for v, a, d, expected in cases:
        result = map_deap_to_class(v, a, d)
        status = "✓" if result == expected else "✗"
        print(f"  {status} V={v} A={a} D={d} → {result} (expected {expected})")

    print("\n=== CREMA-D mapping ===")
    cremad_cases = [
        ("ANG", "angry"),
        ("HAP", "happy"),
        ("SAD", "sad"),
        ("NEU", "calm"),
        ("FEA", "stress"),
        ("DIS", None),
        ("dis", None),    # case-insensitive
    ]
    for raw, expected in cremad_cases:
        result = map_cremad_to_class(raw)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{raw}' → {result} (expected {expected})")
