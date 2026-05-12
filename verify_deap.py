import pickle
import numpy as np
import os

def verify_deap_file(file_path):
    print(f"\n🔍 Checking: {os.path.basename(file_path)}")

    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    print(f"✅ Keys found: {list(data.keys())}")

    d = data['data']
    labels = data['labels']

    print(f"Data shape  : {d.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Data dtype  : {d.dtype}")
    print(f"Labels range: {labels.min():.2f} – {labels.max():.2f}")

    # ---- THE CRITICAL CHECKS ----

    # 1. Shape confirms 128 Hz downsampled, 63s, 40 channels
    if d.shape == (40, 40, 8064):
        print("Shape (40,40,8064) — CORRECT: QMUL preprocessed at 128 Hz")
    elif d.shape[2] > 30000:
        print("Shape looks like RAW data (512 Hz) — wrong file!")
    elif d.shape[2] == 7680:
        print("Shape (40,40,7680) — baseline already trimmed by someone else")
    else:
        print(f"Unexpected shape: {d.shape}")

    # 2. Sampling rate confirmation
    total_seconds = d.shape[2] / 128
    print(f"At 128 Hz → {total_seconds:.1f} seconds per trial (should be 63.0)")

    # 3. EEG amplitude range (preprocessed should be roughly ±150 µV max)
    eeg = d[:, :32, 384:]   # skip 3s baseline
    eeg_max = float(np.abs(eeg).max())
    if eeg_max < 500:
        print(f"EEG amplitude range ±{eeg_max:.1f} µV — looks preprocessed")
    else:
        print(f"EEG amplitude very large ({eeg_max:.1f}) — might be raw/unfiltered")

    # 4. GSR check
    gsr = d[:, 36, 384:]
    print(f"GSR channel 36 — mean: {gsr.mean():.4f}, min: {gsr.min():.4f}, max: {gsr.max():.4f}")
    if gsr.min() >= 0:
        print("GSR is positive — consistent with EDA/skin conductance")
    else:
        print("GSR has negative values — unusual, check channel index")

    # 5. Labels range
    if labels.min() >= 1.0 and labels.max() <= 9.0:
        print(f"Labels in [1–9] — correct DEAP rating scale")
    else:
        print(f"Labels out of expected range: {labels.min():.2f}–{labels.max():.2f}")

    print(f"\nSample EEG[0,0,0:5]: {d[0, 0, 0:5]}")


if __name__ == "__main__":
    data_folder = r"C:\Desktop\FYP\V2 MedOracle\MedOracle\data\DEAP"

    print(f"Looking in: {data_folder}\n")

    if not os.path.exists(data_folder):
        print("Folder not found!")
    else:
        file_path = os.path.join(data_folder, "s01.dat")
        if os.path.exists(file_path):
            verify_deap_file(file_path)
            files = [f for f in os.listdir(data_folder) if f.endswith('.dat')]
            print(f"\nFound {len(files)} subject files (need 32)")
            if len(files) == 32:
                print("All 32 subjects present")
            else:
                print(f"Expected 32, got {len(files)}")
        else:
            print(f"s01.dat not found in {data_folder}")


"""
Issue 1 — Labels minimum is 0, not 1
Issue 2 — GSR values are NOT in µS
"""