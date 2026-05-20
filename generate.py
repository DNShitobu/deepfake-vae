"""Beta-VAE — Inference & Latent Exploration Script"""
import argparse, torch
from torchvision.utils import save_image
from model import BetaVAE

def load(ckpt, latent_dim=512, beta=4.0, device="cpu"):
    m = BetaVAE(latent_dim, beta).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    m.eval()
    return m

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mode", default="sample", choices=["sample","interpolate","traverse"])
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--out", default="vae_output.png")
    p.add_argument("--latent_dim", type=int, default=512)
    p.add_argument("--beta", type=float, default=4.0)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(args.ckpt, args.latent_dim, args.beta, device)
    with torch.no_grad():
        if args.mode == "sample":
            imgs = model.sample(args.n, device)
        elif args.mode == "interpolate":
            z1 = torch.randn(1, args.latent_dim, device=device)
            z2 = torch.randn(1, args.latent_dim, device=device)
            alphas = torch.linspace(0, 1, args.n, device=device)
            imgs = torch.cat([model.decode(z1*(1-a)+z2*a) for a in alphas])
        elif args.mode == "traverse":
            z = torch.zeros(1, args.latent_dim, device=device)
            imgs = []
            for dim in range(min(args.n, args.latent_dim)):
                for v in torch.linspace(-3, 3, 8, device=device):
                    zc = z.clone(); zc[0, dim] = v
                    imgs.append(model.decode(zc))
            imgs = torch.cat(imgs)
    save_image(imgs * 0.5 + 0.5, args.out, nrow=8)
    print(f"Saved to {args.out}")
