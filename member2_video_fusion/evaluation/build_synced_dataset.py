"""
member2_video_fusion/evaluation/build_synced_dataset.py
=======================================================
Build a SYNCHRONISED cross-dataset sample set for multimodal testing / demo.

The problem: no real subject has EEG + GSR + video together — DEAP has
physiological signals (different people) and CREMA-D has facial video (different
people). To exercise and demonstrate the multimodal pipeline we construct
"virtual subjects" by pairing, **by shared emotion label**, a real DEAP EEG/GSR
window with a real CREMA-D video clip. Both modalities genuinely express the
same emotion, even though they come from different real people — the standard
construction for evaluating cross-dataset late fusion.

Output (one self-contained folder per virtual subject):
    data/synced_samples/
        subject_01/  eeg.npy (32,512)  gsr.npy (512,)  video.flv   meta.json
        ...
        subject_10/
        manifest.json

Default: 5 emotions × 2 subjects = 10 virtual subjects.

Channel layout / windowing match training (deap_loader.py):
    EEG = channels 0–31   GSR = channel 36   3 s baseline trimmed   4 s windows.

Run (needs data/DEAP/*.dat and data/CREMA-D/{manifest.csv, VideoFlash/}):
    python -m member2_video_fusion.evaluation.build_synced_dataset
"""

from __future__ import annotations

import csv
import json
import pickle
import random
import shutil
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.label_harmonization import map_deap_to_class

DEAP_DIR        = _ROOT / "data" / "DEAP"
CREMAD_MANIFEST = _ROOT / "data" / "CREMA-D" / "manifest.csv"
OUT_DIR         = _ROOT / "data" / "synced_samples"

EMOTIONS         = ["stress", "calm", "happy", "sad", "angry"]
N_PER_EMOTION    = 2
BASELINE_SAMPLES = 384
WINDOW_SAMPLES   = 512
GSR_CHANNEL      = 36
WINDOW_IDX       = 7
SEED             = 42


def _collect_deap() -> dict[str, list]:
    """emotion -> candidates (clearest first): (margin, subj, trial, v,a,d, eeg, gsr)."""
    cand: dict[str, list] = {e: [] for e in EMOTIONS}
    for dat in sorted(DEAP_DIR.glob("s*.dat")):
        with open(dat, "rb") as f:
            raw = pickle.load(f, encoding="latin1")
        data, labels = raw["data"], raw["labels"]
        for t in range(data.shape[0]):
            v, a, d = float(labels[t, 0]), float(labels[t, 1]), float(labels[t, 2])
            vc, ac, dc = (min(9.0, max(1.0, v)), min(9.0, max(1.0, a)), min(9.0, max(1.0, d)))
            emotion = map_deap_to_class(vc, ac, dc)
            if emotion is None:
                continue
            margin = min(abs(vc - 5.0), abs(ac - 5.0), abs(dc - 5.0))
            s = WINDOW_IDX * WINDOW_SAMPLES
            trial = data[t, :, BASELINE_SAMPLES:]
            eeg = trial[:32, s:s + WINDOW_SAMPLES].astype(np.float32)
            gsr = trial[GSR_CHANNEL, s:s + WINDOW_SAMPLES].astype(np.float32)
            cand[emotion].append((margin, dat.stem, t, v, a, d, eeg, gsr))
    for e in EMOTIONS:
        cand[e].sort(key=lambda x: -x[0])
    return cand


def _collect_cremad() -> dict[str, list]:
    """emotion -> shuffled clip rows (HI-intensity preferred)."""
    with open(CREMAD_MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    by_emo: dict[str, list] = {e: [] for e in EMOTIONS}
    for r in rows:
        if r["emotion"] in by_emo:
            by_emo[r["emotion"]].append(r)
    rng = random.Random(SEED)
    for e in EMOTIONS:
        clips = by_emo[e]
        hi = [c for c in clips if c.get("intensity") == "HI"]
        pool = hi if len(hi) >= N_PER_EMOTION else clips
        rng.shuffle(pool)
        by_emo[e] = pool
    return by_emo


def main() -> None:
    for p in (DEAP_DIR, CREMAD_MANIFEST):
        if not p.exists():
            raise FileNotFoundError(f"Required data not found: {p}")

    deap = _collect_deap()
    cremad = _collect_cremad()

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    manifest = []
    subj_i = 0
    print(f"Building synced subjects ({N_PER_EMOTION} per emotion)…\n")

    for emotion in EMOTIONS:
        for k in range(N_PER_EMOTION):
            subj_i += 1
            margin, dsubj, dtrial, v, a, d, eeg, gsr = deap[emotion][k]
            clip = cremad[emotion][k]

            subj_dir = OUT_DIR / f"subject_{subj_i:02d}"
            subj_dir.mkdir()
            np.save(subj_dir / "eeg.npy", eeg)
            np.save(subj_dir / "gsr.npy", gsr)
            src_video = _ROOT / clip["path"]
            vext = src_video.suffix
            shutil.copy(src_video, subj_dir / f"video{vext}")

            meta = {
                "subject":       f"subject_{subj_i:02d}",
                "emotion":       emotion,
                "deap_source":   {"subject": dsubj, "trial": dtrial,
                                  "valence": v, "arousal": a, "dominance": d},
                "cremad_source": {"file": clip["filename"], "actor": clip["actor_id"],
                                  "intensity": clip.get("intensity", "")},
                "files":         {"eeg": "eeg.npy", "gsr": "gsr.npy", "video": f"video{vext}"},
            }
            with open(subj_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            manifest.append(meta)
            print(f"  subject_{subj_i:02d}: {emotion:7s}  "
                  f"DEAP {dsubj} t{dtrial:<2d}  +  CREMA-D {clip['filename']}")

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump({"n_subjects": subj_i, "emotions": EMOTIONS, "subjects": manifest}, f, indent=2)

    print(f"\n✓ {subj_i} synchronised subjects written to {OUT_DIR}")
    print("  Each has eeg.npy + gsr.npy + video — upload the three files in the web app.")


if __name__ == "__main__":
    main()
