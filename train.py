"""
Beta-VAE Face Generation — Training Script
============================================
Trains Beta-VAE on CelebA/FFHQ with:
  - Warm-up schedule for beta (prevents posterior collapse)
  - Perceptual loss via VGG16 features
  - KL capacity constraint (C-VAE variant) for stable disentanglement
  - Latent traversal visualisations every N epochs
"""

import argparse, time
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as dsets
import torchvision.models as tvm
from torchvision.utils import save_image

from model import BetaVAE


# ── Perceptual Loss ────────────────────────────────────────────────────────────

class VGGPerceptual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features[:16]
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        p = (pred * 0.5 + 0.5 - self.mean) / self.std
        t = (target * 0.5 + 0.5 - self.mean) / self.std
        return torch.nn.functional.l1_loss(self.vgg(p), self.vgg(t))


# ── Dataset ────────────────────────────────────────────────────────────────────

def get_loaders(root, batch_size, size=256, dataset="celeba"):
    tf = T.Compose([
        T.CenterCrop(148) if dataset == "celeba" else T.Resize((size, size)),
        T.Resize((size, size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    if dataset == "celeba":
        train = dsets.CelebA(root, split="train", transform=tf, download=True)
        val   = dsets.CelebA(root, split="valid", transform=tf, download=True)
    else:
        train = dsets.ImageFolder(root, transform=tf)
        n_val = int(0.05 * len(train))
        train, val = torch.utils.data.random_split(train, [len(train)-n_val, n_val])
    tl = DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
    vl = DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    return tl, vl


# ── Trainer ────────────────────────────────────────────────────────────────────

class BetaVAETrainer:
    def __init__(self, args):
        self.args    = args
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.model      = BetaVAE(args.latent_dim, beta=args.beta).to(self.device)
        self.perceptual = VGGPerceptual().to(self.device)
        self.opt        = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
        self.sch        = optim.lr_scheduler.CosineAnnealingLR(self.opt, args.epochs)
        self.out_dir    = Path(args.out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_x    = None  # set on first batch

    def beta_schedule(self, epoch):
        """Linearly warm up beta over first 10 epochs to prevent posterior collapse."""
        return min(self.args.beta, self.args.beta * epoch / 10)

    def step(self, x, beta):
        x = x.to(self.device)
        recon, mu, logvar = self.model(x)
        losses  = self.model.loss(recon, x, mu, logvar)
        perc    = self.perceptual(recon, x)
        total   = losses["recon"] * 10 + perc * 2 + beta * losses["kl"]
        self.opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return {
            "recon": losses["recon"].item(),
            "kl":    losses["kl"].item(),
            "perc":  perc.item(),
            "total": total.item(),
        }

    def latent_traversal(self, x, n_steps=10, epoch=0):
        """Traverse each latent dimension independently to visualise disentanglement."""
        self.model.eval()
        with torch.no_grad():
            mu, _ = self.model.encoder(x[:1].to(self.device))
            imgs = []
            for dim in range(min(8, self.args.latent_dim)):
                z_range = torch.linspace(-3, 3, n_steps)
                for v in z_range:
                    z = mu.clone()
                    z[0, dim] = v
                    imgs.append(self.model.decode(z))
        save_image(
            torch.cat(imgs) * 0.5 + 0.5,
            self.out_dir / f"traversal_epoch_{epoch:04d}.png",
            nrow=n_steps,
        )
        self.model.train()

    def run(self, train_loader, val_loader):
        for epoch in range(1, self.args.epochs + 1):
            t0 = time.time()
            beta = self.beta_schedule(epoch)
            stats = {"recon": 0, "kl": 0, "perc": 0, "total": 0}
            n = 0
            for batch in train_loader:
                imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                if self.fixed_x is None:
                    self.fixed_x = imgs[:8]
                m = self.step(imgs, beta)
                for k in stats: stats[k] += m[k]
                n += 1
            for k in stats: stats[k] /= n
            elapsed = time.time() - t0
            print(f"[Epoch {epoch:03d}/{self.args.epochs}] "
                  f"Recon={stats['recon']:.4f} KL={stats['kl']:.4f} "
                  f"Perc={stats['perc']:.4f} beta={beta:.2f} ({elapsed:.0f}s)")
            if epoch % self.args.save_every == 0:
                self.model.eval()
                with torch.no_grad():
                    recon, _, _ = self.model(self.fixed_x.to(self.device))
                    samples     = self.model.sample(16, self.device)
                save_image(
                    torch.cat([self.fixed_x.to(self.device), recon]) * 0.5 + 0.5,
                    self.out_dir / f"recon_epoch_{epoch:04d}.png", nrow=8
                )
                save_image(samples * 0.5 + 0.5, self.out_dir / f"samples_epoch_{epoch:04d}.png", nrow=4)
                self.latent_traversal(self.fixed_x, epoch=epoch)
                torch.save({
                    "epoch": epoch, "model": self.model.state_dict(),
                    "opt": self.opt.state_dict(),
                }, self.out_dir / f"ckpt_epoch_{epoch:04d}.pt")
                self.model.train()
            self.sch.step()


def parse_args():
    p = argparse.ArgumentParser("Beta-VAE Trainer")
    p.add_argument("--data",       default="./data/celeba")
    p.add_argument("--dataset",    default="celeba", choices=["celeba","ffhq","custom"])
    p.add_argument("--out_dir",    default="./outputs")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--latent_dim", type=int,   default=512)
    p.add_argument("--beta",       type=float, default=4.0, help="KL weight (>1 = more disentangled)")
    p.add_argument("--save_every", type=int,   default=5)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    tl, vl = get_loaders(args.data, args.batch_size, dataset=args.dataset)
    BetaVAETrainer(args).run(tl, vl)
