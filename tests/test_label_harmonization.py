"""
tests/test_label_harmonization.py
==================================
Full test suite for shared/label_harmonization.py

Run with Python's built-in unittest (no external dependencies):
    python3 -m unittest tests.test_label_harmonization -v

Or, if pytest is installed:
    pytest tests/test_label_harmonization.py -v

All tests must pass before any team member begins dataset loading.
"""

import sys
import os
import unittest

# Allow import from project root (run from FYP/ directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.label_harmonization import (
    EMOTION_CLASSES,
    map_deap_to_class,
    map_deap_to_class_int,
    map_cremad_to_class,
    map_cremad_to_class_int,
    batch_map_deap,
    batch_map_cremad,
)


# ===========================================================================
# 1. EMOTION_CLASSES constant
# ===========================================================================

class TestEmotionClassesConstant(unittest.TestCase):

    def test_has_exactly_five_classes(self):
        self.assertEqual(len(EMOTION_CLASSES), 5)

    def test_keys_are_correct(self):
        self.assertEqual(set(EMOTION_CLASSES.keys()), {"stress", "calm", "happy", "sad", "angry"})

    def test_values_are_0_to_4(self):
        self.assertEqual(set(EMOTION_CLASSES.values()), {0, 1, 2, 3, 4})

    def test_specific_indices(self):
        self.assertEqual(EMOTION_CLASSES["stress"], 0)
        self.assertEqual(EMOTION_CLASSES["calm"],   1)
        self.assertEqual(EMOTION_CLASSES["happy"],  2)
        self.assertEqual(EMOTION_CLASSES["sad"],    3)
        self.assertEqual(EMOTION_CLASSES["angry"],  4)


# ===========================================================================
# 2. map_deap_to_class — canonical cases
# ===========================================================================

class TestDeapCanonicalCases(unittest.TestCase):
    """Each of the 5 emotion classes maps from a clear V/A/D triple."""

    def test_stress(self):
        self.assertEqual(map_deap_to_class(3.0, 7.0, 2.0), "stress")

    def test_calm(self):
        self.assertEqual(map_deap_to_class(7.0, 3.0, 8.0), "calm")

    def test_happy(self):
        self.assertEqual(map_deap_to_class(7.0, 7.0, 7.0), "happy")

    def test_sad(self):
        self.assertEqual(map_deap_to_class(2.0, 2.0, 2.0), "sad")

    def test_angry(self):
        self.assertEqual(map_deap_to_class(2.0, 7.0, 7.0), "angry")


# ===========================================================================
# 3. map_deap_to_class — boundary values (exactly == 5)
# ===========================================================================

class TestDeapBoundaryValues(unittest.TestCase):
    """Boundary rule: values == 5 treated as >= 5 (positive side)."""

    def test_all_at_threshold_is_happy(self):
        # V=5 (hi), A=5 (hi), D=5 (hi) → happy
        self.assertEqual(map_deap_to_class(5.0, 5.0, 5.0), "happy")

    def test_v_hi_a_hi_d_lo_is_happy(self):
        # V=5 (hi), A=5 (hi), D=4.9 (lo) → happy
        # Russell (1980) 2D rule: high valence + high arousal = happy,
        # independent of dominance (Mehrabian's PAD axis). Recovers samples
        # that were previously unclassified.
        self.assertEqual(map_deap_to_class(5.0, 5.0, 4.9), "happy")

    def test_v_lo_a_hi_d_hi_is_angry(self):
        # V=4.9 (lo), A=5 (hi), D=5 (hi) → angry
        self.assertEqual(map_deap_to_class(4.9, 5.0, 5.0), "angry")

    def test_v_lo_a_hi_d_lo_is_stress(self):
        # V=4.9 (lo), A=5 (hi), D=4.9 (lo) → stress
        self.assertEqual(map_deap_to_class(4.9, 5.0, 4.9), "stress")

    def test_v_hi_a_lo_d_hi_is_calm(self):
        # V=5 (hi), A=4.9 (lo), D=5 (hi) → calm
        self.assertEqual(map_deap_to_class(5.0, 4.9, 5.0), "calm")

    def test_v_lo_a_lo_d_hi_is_unclassifiable(self):
        # V=4.9 (lo), A=4.9 (lo), D=5 (hi) → no canonical class
        self.assertIsNone(map_deap_to_class(4.9, 4.9, 5.0))

    def test_exactly_1_is_valid_sad(self):
        self.assertEqual(map_deap_to_class(1.0, 1.0, 1.0), "sad")

    def test_exactly_9_is_valid_happy(self):
        self.assertEqual(map_deap_to_class(9.0, 9.0, 9.0), "happy")


# ===========================================================================
# 4. map_deap_to_class — unclassifiable combinations
# ===========================================================================

class TestDeapUnclassifiable(unittest.TestCase):
    """One cell in the 2×2×2 V/A/D cube has no canonical mapping.

    (V_hi, A_hi, D_lo now maps to happy via the Russell 2D rule; only
    V_hi, A_lo, D_lo and V_lo, A_lo, D_hi remain unclassifiable.)
    """

    def test_v_hi_a_hi_d_lo_is_happy(self):
        # excited but submissive — Russell 2D: high V + high A → happy
        self.assertEqual(map_deap_to_class(7.0, 7.0, 2.0), "happy")

    def test_v_hi_a_lo_d_lo_returns_none(self):
        # relaxed but submissive — outside the 5-class model
        self.assertIsNone(map_deap_to_class(7.0, 2.0, 2.0))


# ===========================================================================
# 5. map_deap_to_class — out-of-range inputs raise ValueError
# ===========================================================================

class TestDeapRangeValidation(unittest.TestCase):

    def test_valence_below_1_raises(self):
        with self.assertRaises(ValueError) as ctx:
            map_deap_to_class(0.9, 5.0, 5.0)
        self.assertIn("valence", str(ctx.exception))

    def test_valence_above_9_raises(self):
        with self.assertRaises(ValueError) as ctx:
            map_deap_to_class(9.1, 5.0, 5.0)
        self.assertIn("valence", str(ctx.exception))

    def test_arousal_below_1_raises(self):
        with self.assertRaises(ValueError) as ctx:
            map_deap_to_class(5.0, 0.5, 5.0)
        self.assertIn("arousal", str(ctx.exception))

    def test_dominance_above_9_raises(self):
        with self.assertRaises(ValueError) as ctx:
            map_deap_to_class(5.0, 5.0, 9.5)
        self.assertIn("dominance", str(ctx.exception))

    def test_negative_value_raises(self):
        with self.assertRaises(ValueError):
            map_deap_to_class(-1.0, 5.0, 5.0)


# ===========================================================================
# 6. map_deap_to_class_int
# ===========================================================================

class TestDeapToClassInt(unittest.TestCase):

    def test_happy_returns_2(self):
        self.assertEqual(map_deap_to_class_int(7.0, 7.0, 7.0), 2)

    def test_stress_returns_0(self):
        self.assertEqual(map_deap_to_class_int(3.0, 7.0, 2.0), 0)

    def test_unclassifiable_returns_none(self):
        # V_hi, A_lo, D_lo remains unclassifiable (V_hi,A_hi,D_lo now → happy)
        self.assertIsNone(map_deap_to_class_int(7.0, 2.0, 2.0))

    def test_all_emotions_return_correct_index(self):
        cases = {
            "stress": (3.0, 7.0, 2.0),
            "calm":   (7.0, 3.0, 8.0),
            "happy":  (7.0, 7.0, 7.0),
            "sad":    (2.0, 2.0, 2.0),
            "angry":  (2.0, 7.0, 7.0),
        }
        for emotion, (v, a, d) in cases.items():
            with self.subTest(emotion=emotion):
                self.assertEqual(map_deap_to_class_int(v, a, d), EMOTION_CLASSES[emotion])


# ===========================================================================
# 7. map_cremad_to_class — canonical cases
# ===========================================================================

class TestCremaDCanonicalCases(unittest.TestCase):

    def test_ang_maps_to_angry(self):
        self.assertEqual(map_cremad_to_class("ANG"), "angry")

    def test_hap_maps_to_happy(self):
        self.assertEqual(map_cremad_to_class("HAP"), "happy")

    def test_sad_maps_to_sad(self):
        self.assertEqual(map_cremad_to_class("SAD"), "sad")

    def test_neu_maps_to_calm(self):
        self.assertEqual(map_cremad_to_class("NEU"), "calm")

    def test_fea_maps_to_stress(self):
        # Kreibig (2010): fear activates sympathetic system → stress
        self.assertEqual(map_cremad_to_class("FEA"), "stress")

    def test_dis_maps_to_none(self):
        # DIS (disgust) dropped — no clean mapping to 5-class taxonomy
        self.assertIsNone(map_cremad_to_class("DIS"))


# ===========================================================================
# 8. map_cremad_to_class — case insensitivity & whitespace
# ===========================================================================

class TestCremaDCaseInsensitivity(unittest.TestCase):

    def test_lowercase_ang(self):
        self.assertEqual(map_cremad_to_class("ang"), "angry")

    def test_lowercase_dis_returns_none(self):
        self.assertIsNone(map_cremad_to_class("dis"))

    def test_mixed_case_fea(self):
        self.assertEqual(map_cremad_to_class("Fea"), "stress")

    def test_whitespace_stripped(self):
        self.assertEqual(map_cremad_to_class("  HAP  "), "happy")

    def test_all_six_labels_case_insensitive(self):
        for label in ["ANG", "HAP", "SAD", "NEU", "FEA", "DIS"]:
            with self.subTest(label=label):
                self.assertEqual(
                    map_cremad_to_class(label),
                    map_cremad_to_class(label.lower())
                )


# ===========================================================================
# 9. map_cremad_to_class — unknown labels raise ValueError
# ===========================================================================

class TestCremaDUnknownLabel(unittest.TestCase):

    def test_unknown_label_raises(self):
        with self.assertRaises(ValueError) as ctx:
            map_cremad_to_class("JOY")
        self.assertIn("Unknown CREMA-D label", str(ctx.exception))

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            map_cremad_to_class("")

    def test_numeric_string_raises(self):
        with self.assertRaises(ValueError):
            map_cremad_to_class("123")

    def test_partial_label_raises(self):
        with self.assertRaises(ValueError):
            map_cremad_to_class("AN")


# ===========================================================================
# 10. map_cremad_to_class_int
# ===========================================================================

class TestCremaDToClassInt(unittest.TestCase):

    def test_ang_returns_4(self):
        self.assertEqual(map_cremad_to_class_int("ANG"), 4)

    def test_hap_returns_2(self):
        self.assertEqual(map_cremad_to_class_int("HAP"), 2)

    def test_sad_returns_3(self):
        self.assertEqual(map_cremad_to_class_int("SAD"), 3)

    def test_neu_returns_1(self):
        self.assertEqual(map_cremad_to_class_int("NEU"), 1)

    def test_fea_returns_0(self):
        self.assertEqual(map_cremad_to_class_int("FEA"), 0)

    def test_dis_returns_none(self):
        self.assertIsNone(map_cremad_to_class_int("DIS"))

    def test_all_non_dis_labels_return_valid_int(self):
        expected = {"ANG": 4, "HAP": 2, "SAD": 3, "NEU": 1, "FEA": 0}
        for label, idx in expected.items():
            with self.subTest(label=label):
                self.assertEqual(map_cremad_to_class_int(label), idx)


# ===========================================================================
# 11. Batch helpers
# ===========================================================================

class TestBatchHelpers(unittest.TestCase):

    def test_batch_map_deap_basic(self):
        trials = [
            (7.0, 7.0, 7.0),  # happy
            (2.0, 2.0, 2.0),  # sad
            (7.0, 2.0, 2.0),  # unclassifiable (V_hi, A_lo, D_lo)
        ]
        self.assertEqual(batch_map_deap(trials), ["happy", "sad", None])

    def test_batch_map_deap_length_preserved(self):
        trials = [(float(i), float(i), float(i)) for i in range(1, 10)]
        self.assertEqual(len(batch_map_deap(trials)), len(trials))

    def test_batch_map_deap_empty_list(self):
        self.assertEqual(batch_map_deap([]), [])

    def test_batch_map_cremad_basic(self):
        labels   = ["ANG", "HAP", "SAD", "NEU", "FEA", "DIS"]
        expected = ["angry", "happy", "sad", "calm", "stress", None]
        self.assertEqual(batch_map_cremad(labels), expected)

    def test_batch_map_cremad_empty_list(self):
        self.assertEqual(batch_map_cremad([]), [])

    def test_batch_map_cremad_dis_is_none(self):
        results = batch_map_cremad(["ANG", "DIS", "HAP", "DIS"])
        non_none = [r for r in results if r is not None]
        self.assertEqual(non_none, ["angry", "happy"])


# ===========================================================================
# 12. Output integrity — every non-None result is a valid emotion label
# ===========================================================================

class TestOutputIntegrity(unittest.TestCase):

    def test_deap_outputs_are_valid_labels(self):
        cases = [
            (3.0, 7.0, 2.0), (7.0, 3.0, 8.0),
            (7.0, 7.0, 7.0), (2.0, 2.0, 2.0), (2.0, 7.0, 7.0),
        ]
        for v, a, d in cases:
            result = map_deap_to_class(v, a, d)
            if result is not None:
                self.assertIn(result, EMOTION_CLASSES,
                    msg=f"Result '{result}' for V={v} A={a} D={d} is not a valid emotion class")

    def test_cremad_outputs_are_valid_labels(self):
        for label in ["ANG", "HAP", "SAD", "NEU", "FEA"]:
            result = map_cremad_to_class(label)
            self.assertIn(result, EMOTION_CLASSES,
                msg=f"Result '{result}' for '{label}' is not a valid emotion class")

    def test_dis_is_only_none_cremad_label(self):
        none_labels = [
            lbl for lbl in ["ANG", "HAP", "SAD", "NEU", "FEA", "DIS"]
            if map_cremad_to_class(lbl) is None
        ]
        self.assertEqual(none_labels, ["DIS"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
