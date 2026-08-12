#!/usr/bin/env python3
"""Pre-download all Zenodo representation data for tensorpow.

Running this script before benchmarking ensures that network I/O
is completely excluded from the measured execution times.
"""

from tensorpow.file_handler import _ensure_file_downloaded

def main():
    print("Downloading SL(2) (2x2) representations up to n=79...")
    _ensure_file_downloaded("sl2reps.txt")

    print("Downloading SU(3) (3x3) symmetric representations up to degree 30...")
    for k in range(1, 31):
        _ensure_file_downloaded(f"piM_sym_{k}_T_sparse.npz", k=k)
        _ensure_file_downloaded(f"piM_sym_{k}_exps.npz", k=k)
        print(f"Downloaded SU(3) degree {k}/30", end="\r")
        
    print("\nAll precomputed representation data is downloaded and cached successfully.")

if __name__ == "__main__":
    main()
