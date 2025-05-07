import os, csv, argparse, numpy as np, mne, tqdm, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import resample_poly

WINDOW_SEC, STRIDE_SEC, TARGET_FS = 4, 1, 200
MAX_EDF_FILES = 20

def find_annotation(edf_path: str):
    ann = edf_path.replace(".edf", ".seizures")
    if not os.path.exists(ann):
        return []
    with open(ann) as f:
        return [(float(s), float(e)) for s, e in
                (ln.split() for ln in f)]

def overlaps_any(a0, a1, intervals):
    return any(a0 < e and a1 > s for s, e in intervals)

def process_one(edf_path: str, out_root: str):
    raw   = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    src_fs = raw.info["sfreq"]
    decim  = int(round(src_fs / TARGET_FS))

    win_len  = int(WINDOW_SEC * src_fs)
    stride   = int(STRIDE_SEC * src_fs)
    starts   = range(0, int(raw.n_times) - win_len + 1, stride)
    ann      = find_annotation(edf_path)

    out_dir = os.path.join(out_root,
                           os.path.basename(edf_path).removesuffix(".edf"))
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for idx, st in enumerate(starts):
        seg = raw.get_data(start=st, stop=st + win_len, reject_by_annotation=False)

        if abs(src_fs / TARGET_FS - decim) < 1e-3:
            seg = seg[:, ::decim]
        else:
            seg = resample_poly(seg, TARGET_FS, int(src_fs))

        seg = seg.astype(np.float32)
        seg_path = os.path.join(out_dir, f"{idx}.npy")
        np.save(seg_path, seg, allow_pickle=False)

        w0, w1 = st / src_fs, (st + win_len) / src_fs
        label  = int(overlaps_any(w0, w1, ann))
        rows.append([seg_path, label, os.path.basename(edf_path), w0, w1])
    return rows

def gather_edfs(root, require_seizures=False):
    paths = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if not f.endswith(".edf"):
                continue
            p = os.path.join(dp, f)
            if require_seizures and not os.path.exists(p.replace(".edf", ".seizures")):
                continue
            paths.append(p)
    return sorted(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf_dir",  required=True, help="Dir with EDFs")
    ap.add_argument("--out_dir",  default="data/processed")
    ap.add_argument("--meta_csv", default="metadata.csv")
    ap.add_argument("--workers",  type=int, default=min(4, os.cpu_count()),
                    help="Parallel workers (memory bound)")
    args = ap.parse_args()
    edfs_pos = gather_edfs(args.edf_dir, require_seizures=True)
    edfs_neg = gather_edfs(args.edf_dir, require_seizures=False)
    edfs_neg = [p for p in edfs_neg if p not in edfs_pos]
    
    N = 20
    edfs = edfs_pos[:N] + edfs_neg[:N]

    all_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_one, p, args.out_dir): p for p in edfs}
        for fut in tqdm.tqdm(as_completed(futs), total=len(futs), desc="EDF-win"):
            all_rows.extend(fut.result())

    with open(args.meta_csv, "w", newline="") as f:
        csv.writer(f).writerows(
            [["filepath", "label", "recording", "start", "end"]] + all_rows
        )
if __name__ == "__main__":
    main()
