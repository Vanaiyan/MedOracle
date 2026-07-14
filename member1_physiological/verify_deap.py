import pickle
import numpy as np
import os

def verify_deap_file(file_path):
    print(f"\n[CHECK] Checking: {os.path.basename(file_path)}")

    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    print(f"Keys found   : {list(data.keys())}")

    d      = data['data']
    labels = data['labels']

    print(f"Data shape   : {d.shape}")
    print(f"Labels shape : {labels.shape}")
    print(f"Data dtype   : {d.dtype}")
    print(f"Labels range : {labels.min():.2f} to {labels.max():.2f}")

    # ---- CRITICAL CHECKS ----

    # 1. Shape confirms 128 Hz downsampled, 63s, 40 channels
    if d.shape == (40, 40, 8064):
        print("Shape check  : PASS (40,40,8064) -> QMUL preprocessed at 128 Hz")
    elif d.shape[2] > 30000:
        print("Shape check  : FAIL - looks like RAW 512 Hz data (wrong file!)")
    elif d.shape[2] == 7680:
        print("Shape check  : NOTE - baseline already trimmed by uploader")
    else:
        print(f"Shape check  : UNEXPECTED {d.shape}")

    # 2. Sampling rate
    total_seconds = d.shape[2] / 128
    print(f"Sampling rate: {total_seconds:.1f}s per trial at 128 Hz (expect 63.0)")

    # 3. EEG amplitude
    eeg     = d[:, :32, 384:]
    eeg_max = float(np.abs(eeg).max())
    if eeg_max < 500:
        print(f"EEG amplitude: PASS +/-{eeg_max:.1f} uV (looks preprocessed)")
    else:
        print(f"EEG amplitude: WARNING very large ({eeg_max:.1f}) - might be raw")

    # 4. GSR channel 36
    gsr = d[:, 36, 384:]
    print(f"GSR channel 36: mean={gsr.mean():.2f}  min={gsr.min():.2f}  max={gsr.max():.2f}")
    if gsr.min() >= 0:
        print("GSR sign     : positive values only (consistent with EDA)")
    else:
        print("GSR sign     : NOTE - negative values present (raw ADC units, handled by z-score)")

    # 5. Labels range
    if labels.min() >= 1.0 and labels.max() <= 9.0:
        print(f"Labels       : PASS in [1-9] standard DEAP scale")
    else:
        print(f"Labels       : NOTE range {labels.min():.2f} to {labels.max():.2f}")
        print("               (values clipped to [1,9] by deap_loader.py - already handled)")

    print(f"\nSample EEG[0,0,0:5]: {d[0, 0, 0:5]}")
    print("\nVerification complete.")


if __name__ == "__main__":
    data_folder = r"C:\Desktop\FYP\V2 MedOracle - Copy\MedOracle\data\DEAP"

    print(f"Looking in: {data_folder}\n")

    if not os.path.exists(data_folder):
        print("ERROR: Folder not found! Check your path.")
    else:
        file_path = os.path.join(data_folder, "s01.dat")
        if os.path.exists(file_path):
            verify_deap_file(file_path)
            files = [f for f in os.listdir(data_folder) if f.endswith('.dat')]
            print(f"\nSubject files: {len(files)} found (need 32)")
            if len(files) == 32:
                print("All 32 subjects present - READY TO TRAIN")
            else:
                print(f"WARNING: Expected 32, got {len(files)}")
        else:
            print(f"ERROR: s01.dat not found in {data_folder}")
