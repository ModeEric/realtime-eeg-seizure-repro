# Real-Time EEG Seizure Detection — Reproduction of Lee *et al.* (2022)


This repository reproduces the **CNN‑2D + LSTM** baseline from

> K. Lee, H. Jeong, S. Kim *et al.*
> “Real‑Time Seizure Detection Using EEG: A Comprehensive Comparison of Recent Approaches Under a Realistic Setting.” *Proc. AAAI 2022*


## 💡 Key Takeaways

* ✅ Baseline performance is largely recoverable with open data and modest compute.
* ⚠️ Variable channel counts break batching; we pad to 64 leads.
* ⚡ Shorter inference stride halves detection latency at the cost of \~2× compute.

---

## 🚀 Installation

1. **Clone the repo and create a Python environment**

   ```bash
   git clone https://github.com/<your‑org>/eeg‑seizure‑reproduction.git
   cd eeg‑seizure‑reproduction
   python -m venv .venv          # or: conda create -n eeg python=3.10
   source .venv/bin/activate     # on Windows: .\.venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   The list pins `torch`, `mne`, `numpy`, `scipy`, and other utilities tested on Python 3.10/3.11. A CUDA‑enabled PyTorch wheel is picked automatically if a compatible GPU is detected; otherwise the CPU build is installed.

---

## 📦 Dataset

### How to obtain the TUH EEG Seizure Corpus

The Temple University Hospital (TUH) EEG Seizure Corpus is distributed by the Neural Engineering Data Consortium (NEDC) under a free research‑only license. The steps below fetch just the annotated subset we used:

1. **Request access**

   * Download the *“Request to Download the TUH EEG Corpus”* PDF from the NEDC downloads page.
   * Fill it out and email the signed form to **[help@nedcdata.org](mailto:help@nedcdata.org)** with the subject
     `Download The TUH EEG Corpus`.

   Approval usually arrives within 24 h and contains a username (e.g. `nedc‑xxxx`) and password.

2. **Verify `rsync` connectivity**

   `rsync` is pre‑installed on Linux/macOS. Windows users can grab MobaXterm or WSL. Test the credentials with the tiny **TEST** directory:

   ```bash
   rsync -auxvL \
     nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/TEST \
     .
   ```

   The transfer should complete without errors; if it fails, add extra `-v` flags for verbose logs.

3. **Download the seizure‑corpus subset** (≈ 19 GB)

   ```bash
   rsync -auxvL --progress \
     nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_seizure/v1.5.2/edf/dev_test/ \
     ./tusz_v1.5.2/edf/dev_test/
   ```

   * Replace `v1.5.2` with a newer tag (e.g. `v2.0.3`) to grab a more recent release.
   * Omit the version segment entirely to mirror *all* available revisions.

4. **Resume or update later**

   `rsync` is incremental, so re‑running the same command resumes broken downloads or updates the local mirror.
   Add `--partial` to keep unfinished files and `--delete` when you need an exact one‑to‑one mirror.

With the corpus in place, run `python scripts/edf_to_windows.py --root tusz_v1.5.2 ...` to convert EDF recordings into 4‑second, 200 Hz windows and generate the accompanying `metadata.csv`.

---

Happy replicating!
