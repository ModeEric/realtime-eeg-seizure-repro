#!/usr/bin/env python3
import argparse, torch, sklearn.metrics as skm, numpy as np
from torch.utils.data import DataLoader
from models.cnn2d_lstm import CNN2DLSTM
from datasets.windows import WindowDataset
# ---------------------------------------------------------------------
# Helpers for seizure‐interval labels
# ---------------------------------------------------------------------
def find_annotation(edf_path: str):
    """
    TUH ships a companion .seizures file for each EDF, e.g.
      00000002_s002_t000.edf  ->  00000002_s002_t000.seizures
    Each line has two floats: <start_sec> <end_sec>
    Returns list[(start, end)] in seconds (empty list if file missing).
    """
    ann_path = edf_path.replace(".edf", ".seizures")
    intervals = []
    try:
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    intervals.append((float(parts[0]), float(parts[1])))
    except FileNotFoundError:
        pass
    return intervals

def overlaps_any(win_start, win_end, intervals):
    """True if window [win_start, win_end] overlaps any (s,e) in intervals."""
    for s, e in intervals:
        if win_start < e and win_end > s:   # proper overlap test
            return True
    return False

def evaluate(csv, ckpt, subset_frac=0.2, batch=64):
    ds = WindowDataset(csv, subset_frac=subset_frac)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4)
    net = CNN2DLSTM().eval(); net.load_state_dict(torch.load(ckpt, map_location="cpu"))

    preds, labels = [], []
    with torch.no_grad():
        for x,y in dl:
            preds.append(net(x).cpu())
            labels.append(y)
    p = torch.cat(preds).numpy(); y = torch.cat(labels).numpy()
    auroc = skm.roc_auc_score(y, p)
    auprc = skm.average_precision_score(y, p)
    return auroc, auprc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="metadata.csv")
    ap.add_argument("--ckpt", default="ckpt.pth")
    args = ap.parse_args()
    auroc, auprc = evaluate(args.csv, args.ckpt)
    print(f"AUROC={auroc:.3f}  AUPRC={auprc:.3f}")
