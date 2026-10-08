"""Vessel segmentation: a pretrained FCU-Net (third-party, MIT) and a classical fallback that needs no weights."""
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sato
from skimage.morphology import remove_small_objects

from .preprocess import Prepared

WEIGHTS = {  # name -> (url, sha256)
    "drive": ("https://raw.githubusercontent.com/JavierGurrola/FCU-Net/main/pretrained/model_drive.pth",
              "3c28600c965c54cb80ddb992700033fe2cd85b9dfeb53723788d08745d305f07"),
    "all": ("https://raw.githubusercontent.com/JavierGurrola/FCU-Net/main/pretrained/model_all.pth",
            "40258c5a730a04bb1439d4b4db69fa6d0af792d537251ce7be7965e46fc36ae4"),
}
MODEL_KWARGS = {"base filters": 16, "dropout rate": 0.3}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_weights(name: str = "drive", cache_dir="models") -> Path:
    """Download the pretrained weights once (about 1.8 MB) and check their SHA-256."""
    url, digest = WEIGHTS[name]
    path = Path(cache_dir) / f"fcunet_{name}.pth"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    if _sha256(path) != digest:
        path.unlink()
        raise RuntimeError(f"checksum mismatch for {name} weights; the file was removed, try again")
    return path


class FCUNetSegmenter:
    """Probability map of 'vessel' for a prepared fundus image (needs `pip install -e ".[fcunet]"`)."""

    def __init__(self, weights: str = "drive", cache_dir="models", device: str | None = None, tta: bool = False):
        import torch
        from .fcunet import FCUnet
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tta = tta
        self.model = FCUnet(**MODEL_KWARGS).to(self.device)
        state = torch.load(fetch_weights(weights, cache_dir), map_location=self.device)
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()

    def _forward(self, x):
        with self.torch.no_grad():
            return self.torch.sigmoid(self.model(x))

    def probability(self, img: np.ndarray) -> np.ndarray:
        """img: HxWx3 uint8 (CLAHE-enhanced); H and W must be multiples of 8."""
        x = self.torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if not self.tta:
            return self._forward(x)[0, 0].cpu().numpy()
        outs = []   # 8 flips / rotations, undone and averaged (the authors' test-time ensemble)
        for k in range(4):
            for flip in (False, True):
                xt = self.torch.rot90(x, k, (2, 3))
                xt = self.torch.flip(xt, (3,)) if flip else xt
                y = self._forward(xt)
                y = self.torch.flip(y, (3,)) if flip else y
                outs.append(self.torch.rot90(y, -k, (2, 3)))
        return self.torch.stack(outs).mean(0)[0, 0].cpu().numpy()

    def segment(self, prep: Prepared, threshold: float = 0.5) -> np.ndarray:
        return clean_vessels(self.probability(prep.enhanced) > threshold, prep.fov)


def clean_vessels(mask: np.ndarray, fov: np.ndarray, border: int = 6, min_size: int = 30) -> np.ndarray:
    """Keep vessels inside the retina (minus a thin rim, where the network reacts to the edge) and drop specks."""
    inner = ndi.binary_erosion(fov, iterations=border)
    return remove_small_objects(mask & inner, min_size=min_size, connectivity=2)


def classical_vessels(prep: Prepared, sigmas=(1.5, 2.0, 3.0), keep: float = 0.12) -> np.ndarray:
    """Weight-free fallback: multi-scale ridge filter (Sato) on the green channel; the strongest `keep` fraction of
    the retina is called vessel. Cruder than the network, but deterministic and dependency-light (used in the tests)."""
    g = prep.enhanced[..., 1].astype(np.float32) / 255.0
    resp = sato(g, sigmas=sigmas, black_ridges=True)
    inner = ndi.binary_erosion(prep.fov, iterations=6)
    thr = np.quantile(resp[inner], 1.0 - keep)
    return clean_vessels(resp > thr, prep.fov)
