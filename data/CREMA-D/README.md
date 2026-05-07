# CREMA-D Dataset — Download Instructions

**Used by**: Member 2 (Vanaiyan) — Video + Fusion Module  
**DO NOT commit raw video/audio files to Git** (they are in `.gitignore`)

---

## About CREMA-D

- 91 actors (48 male, 43 female), ages 20–74
- ~7,442 video clips at ~30 fps
- 6 original emotion labels: ANG, DIS, FEA, HAP, NEU, SAD

Reference: Cao et al. (2014), IEEE Trans. Affective Computing

---

## MedOracle Label Mapping

| CREMA-D | → MedOracle class | Notes |
|---------|-------------------|-------|
| ANG     | angry             | Direct match |
| HAP     | happy             | Direct match |
| SAD     | sad               | Direct match |
| NEU     | calm              | Neutral ≈ calm |
| FEA     | stress            | Kreibig (2010): fear → sympathetic activation |
| DIS     | **DROPPED**       | No clean mapping — exclude from training |

---

## How to Download

### Option A — GitHub (recommended, ~4 GB)

```bash
# Clone the full CREMA-D repository (videos in AudioWAV + VideoMP4)
git clone https://github.com/CheyneyComputerScience/CREMA-D.git data/CREMA-D/raw/

# VideoMP4 is ~2.5 GB. AudioWAV is ~1.5 GB.
# You only need VideoMP4 for Member 2's pipeline.
```

### Option B — Direct download

Go to: https://github.com/CheyneyComputerScience/CREMA-D  
Download the `VideoMP4/` folder from the repository.

---

## Expected File Structure After Download

```
data/CREMA-D/
├── README.md                   ← this file
└── raw/
    ├── VideoMP4/
    │   ├── 1001_DFA_ANG_XX.mp4   ← ActorID_Sentence_Emotion_Intensity
    │   ├── 1001_DFA_DIS_XX.mp4
    │   ├── ...
    │   └── 1091_WSI_SAD_HI.mp4
    └── processedResults/
        └── summaryTable.csv      ← ground-truth labels + metadata
```

---

## Filename Convention

```
{ActorID}_{SentenceID}_{EmotionLabel}_{Intensity}.mp4

Examples:
  1001_DFA_ANG_XX.mp4   → Actor 1001, sentence DFA, ANGRY, unknown intensity
  1045_IEO_FEA_HI.mp4   → Actor 1045, sentence IEO, FEAR, high intensity
  1091_WSI_SAD_LO.mp4   → Actor 1091, sentence WSI, SAD, low intensity

EmotionLabel codes: ANG DIS FEA HAP NEU SAD
Intensity codes:    LO (low)  MD (medium)  HI (high)  XX (unspecified)
```

---

## Parsing Filenames + Label Harmonization

```python
import os
from pathlib import Path
from shared.label_harmonization import map_cremad_to_class

video_dir = Path("data/CREMA-D/raw/VideoMP4")
records = []

for mp4_path in sorted(video_dir.glob("*.mp4")):
    stem = mp4_path.stem                        # e.g. "1001_DFA_ANG_XX"
    parts = stem.split("_")
    actor_id      = int(parts[0])               # e.g. 1001
    emotion_code  = parts[2]                    # e.g. "ANG"
    
    emotion_label = map_cremad_to_class(emotion_code)
    if emotion_label is None:
        continue                                # Drop DIS samples
    
    records.append({
        "path":     str(mp4_path),
        "actor_id": actor_id,
        "emotion":  emotion_label,
    })

print(f"Loaded {len(records)} clips after dropping DIS")
# Expect ~6,250–6,500 clips (original 7,442 minus ~1,100 DIS clips)
```

---

## Verification Checklist

After downloading, run this quick check:

```python
from pathlib import Path

video_dir = Path("data/CREMA-D/raw/VideoMP4")
mp4_files = list(video_dir.glob("*.mp4"))
assert len(mp4_files) >= 7000, f"Expected ~7,442 clips, found {len(mp4_files)}"

# Check actor count
actor_ids = set(int(f.stem.split("_")[0]) for f in mp4_files)
assert len(actor_ids) == 91, f"Expected 91 actors, found {len(actor_ids)}"

print(f"✓ {len(mp4_files)} video clips, {len(actor_ids)} actors")
```
