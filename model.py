"""
Beta-VAE Face Generation — Model Architecture
===============================================
Variational Autoencoder with disentangled latent space for face generation.

Architecture:
  Encoder : ResNet encoder -> (mu, logvar) in latent space
  Decoder : latent z -> reconstructed face (symmetric to encoder)

Loss:
  ELBO = E[log p(x|z)] - beta * KL(q(z|x) || p(z))
  where:
    Reconstruction term = perceptual loss + L1
    KL term controls disentanglement (beta > 1 encourages factored representations)

Key properties:
  - Smooth, traversable latent space (unlike GANs)
  - Each dimension encodes a disentangled facial attribute (pose, lighting, expression)
  - Generates new faces by sampling z ~ N(0,I) and decoding
  - Interpolation between faces by blending latent codes

Reference:
  Higgins et al. (2017) "beta-VAE: Learning Basic Visual Concepts"
  Kingma & Welling (2013) "Auto-Encoding Variational Bayes"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.GroupNorm(8, ch),
        )
    def forward(self, x):
        return F.silu(x + self.net(x))


class AttentionBlock(nn.Module):
    """Self-attention at bottleneck to capture global structure."""
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.reshape(B, C, H*W).permute(0, 2, 1)
        k = k.reshape(B, C, H*W)
        v = v.reshape(B, C, H*W).permute(0, 2, 1)
        attn = F.softmax(q @ k / C**0.5, dim=-1)
        h = (attn @ v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(h)


# ── Encoder ────────────────────────────────────────────────────────────────────

class VAEEncoder(nn.Module):
    """
    256x256 face -> (mu, logvar) pair in R^latent_dim
    Outputs parameters of the approximate posterior q(z|x)
    """
    def __init__(self, in_ch=3, latent_dim=512, ch_mult=(1, 2, 4, 8)):
        super().__init__()
        base_ch = 64
        chs = [base_ch * m for m in ch_mult]  # [64, 128, 256, 512]

        self.stem = nn.Conv2d(in_ch, base_ch, 3, 1, 1)
        self.downs = nn.ModuleList()
        prev = base_ch
        for ch in chs:
            self.downs.append(nn.Sequential(
                ResBlock(prev),
                nn.Conv2d(prev, ch, 4, 2, 1),   # 2x downsample
                nn.GroupNorm(8, ch), nn.SiLU(),
            ))
            prev = ch
        # 16x16 spatial at bottleneck
        self.mid = nn.Sequential(ResBlock(prev), AttentionBlock(prev), ResBlock(prev))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu     = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

    def forward(self, x):
        x = self.stem(x)
        for down in self.downs:
            x = down(x)
        x = self.mid(x)
        h = self.pool(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


# ── Decoder ────────────────────────────────────────────────────────────────────

class VAEDecoder(nn.Module):
    """
    z in R^latent_dim -> 256x256 face
    Symmetric to encoder with progressive upsampling
    """
    def __init__(self, out_ch=3, latent_dim=512, ch_mult=(8, 4, 2, 1)):
        super().__init__()
        base_ch = 64
        chs = [base_ch * m for m in ch_mult]   # [512, 256, 128, 64]
        self.fc = nn.Linear(latent_dim, chs[0] * 8 * 8)
        self.ups = nn.ModuleList()
        for i, (inc, outc) in enumerate(zip(chs, chs[1:] + [base_ch])):
            self.ups.append(nn.Sequential(
                ResBlock(inc),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(inc, outc, 3, 1, 1),
                nn.GroupNorm(8, outc), nn.SiLU(),
            ))
        self.out = nn.Sequential(
            ResBlock(base_ch),
            nn.Conv2d(base_ch, out_ch, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 512, 8, 8)
        for up in self.ups:
            x = up(x)
        return self.out(x)


# ── Beta-VAE ──────────────────────────────────────────────────────────────────

class BetaVAE(nn.Module):
    """
    Full Beta-VAE model.
    
    Usage:
        model = BetaVAE(latent_dim=512, beta=4.0)
        recon, mu, logvar = model(face_batch)
        loss = model.loss(recon, face_batch, mu, logvar)
        
        # Generate new face
        z = torch.randn(1, 512)
        new_face = model.decode(z)
        
        # Interpolate between two faces
        z1, z2 = model.encode(face_a), model.encode(face_b)
        z_interp = 0.5 * z1 + 0.5 * z2
        interp_face = model.decode(z_interp)
    """
    def __init__(self, latent_dim: int = 512, beta: float = 4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta       = beta
        self.encoder    = VAEEncoder(latent_dim=latent_dim)
        self.decoder    = VAEDecoder(latent_dim=latent_dim)

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick: z = mu + eps * sigma, eps ~ N(0,I)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at eval time

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def sample(self, n: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)

    def loss(self, recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor) -> dict:
        """
        ELBO = reconstruction loss + beta * KL divergence
        Returns dict with total loss and individual components.
        """
        # Reconstruction: L1 is more stable than MSE for faces
        recon_loss = F.l1_loss(recon, x, reduction="mean")
        # KL: sum over latent dims, mean over batch
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total   = recon_loss + self.beta * kl_loss
        return {"loss": total, "recon": recon_loss, "kl": kl_loss}


if __name__ == "__main__":
    model = BetaVAE(latent_dim=512, beta=4.0)
    x = torch.randn(2, 3, 256, 256)
    recon, mu, logvar = model(x)
    losses = model.loss(recon, x, mu, logvar)
    print(f"Recon: {recon.shape}")
    print(f"Losses: {losses}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {n_params:.2f}M")
