# Vessel network weights

`retina-extract` downloads the pretrained FCU-Net weights here on first use (about 1.8 MB) and checks their SHA-256:

| File | Source | SHA-256 |
|---|---|---|
| `fcunet_drive.pth` | [JavierGurrola/FCU-Net](https://github.com/JavierGurrola/FCU-Net) (`pretrained/model_drive.pth`, MIT) | `3c28600c965c54cb80ddb992700033fe2cd85b9dfeb53723788d08745d305f07` |
| `fcunet_all.pth` | same repository (`pretrained/model_all.pth`) | `40258c5a730a04bb1439d4b4db69fa6d0af792d537251ce7be7965e46fc36ae4` |

The network was trained on public vessel-segmentation datasets (DRIVE, STARE, CHASE), not on the images used here.
