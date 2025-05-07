import numpy as np, torch
from torch.utils.data import Dataset

N_CH = 64

class WindowDataset(Dataset):
    def __init__(self, csv_path, subset_frac=1.0):
        rows = np.loadtxt(csv_path, dtype=str, delimiter=",", skiprows=1)
        if subset_frac < 1.0:
            rows = rows[:int(len(rows)*subset_frac)]
        self.paths  = rows[:,0]
        self.labels = rows[:,1].astype(np.float32)

    def __len__(self):  return len(self.paths)

    def _fix_channels(self, x: np.ndarray):
        c, t = x.shape
        if c == N_CH:
            return x
        elif c < N_CH:
            pad = np.zeros((N_CH - c, t), dtype=x.dtype)
            return np.concatenate([x, pad], axis=0)
        else:
            return x[:N_CH]

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])
        x = self._fix_channels(x)
        return torch.tensor(x), torch.tensor(self.labels[idx])
