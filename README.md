# Deepfake Beta-VAE Face Generation

Beta-Variational Autoencoder for disentangled face generation. Part of MPhil research on hybrid multi-modal deepfake detection.

## Architecture
- Encoder: ResNet-style CNN + Self-Attention at bottleneck -> (mu, logvar)
- Decoder: Progressive upsample with GroupNorm + SiLU
- Loss: Reconstruction (L1) + Perceptual (VGG16) + beta*KL

## Key Research Properties
- **Disentangled latent space**: each dimension encodes independent factors (pose, expression, lighting)
- **Smooth interpolation**: blend faces by interpolating in latent space
- **Latent traversal**: visualise what each dimension controls
- **No mode collapse**: stable training unlike GANs

## VAE Artifacts for Detection Research
- Blurriness (over-smoothing from MSE/L1 reconstruction)
- Loss of high-frequency texture detail
- Posterior collapse in unused latent dimensions
- Systematic underrepresentation of rare attributes

## Training
```bash
python train.py --dataset celeba --data ./data --epochs 100 --beta 4.0
```

## Generation
```bash
python generate.py --ckpt outputs/ckpt_epoch_0100.pt --mode sample --n 16
python generate.py --ckpt outputs/ckpt_epoch_0100.pt --mode interpolate --n 10
python generate.py --ckpt outputs/ckpt_epoch_0100.pt --mode traverse --n 8
```

---
MPhil Research | [Dnshitobu](https://github.com/Dnshitobu)
