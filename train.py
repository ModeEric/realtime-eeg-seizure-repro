#!/usr/bin/env python3
import argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader
from models.cnn2d_lstm import CNN2DLSTM
from datasets.windows import WindowDataset
from tqdm import tqdm
def train_epoch(model, loader, device):
    model.train(); loss_fn = nn.BCELoss(); opt = torch.optim.AdamW(model.parameters(), 1e-4)
    running = 0
    for x,y in tqdm(loader):
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        preds = model(x)
        loss  = loss_fn(preds, y)
        loss.backward(); opt.step()
        running += loss.item()*x.size(0)
    return running/len(loader.dataset)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="metadata.csv")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--subset_frac", type=float, default=0.2)  # small dev run
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--ckpt", default="ckpt.pth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds  = WindowDataset(args.csv, subset_frac=args.subset_frac)
    dl  = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    net = CNN2DLSTM().to(device)

    for ep in range(args.epochs):
        loss = train_epoch(net, dl, device)
        print(f"Epoch {ep+1}  loss={loss:.4f}")

    torch.save(net.state_dict(), args.ckpt)
    print("Checkpoint saved →", args.ckpt)

if __name__ == "__main__":
    main()
