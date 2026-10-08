# Data (not versioned)

The images come from the Kaggle **APTOS 2019 Blindness Detection** competition (subject to its terms; not redistributed here).

1. Download the training images and `train.csv` (columns `id_code`, `diagnosis` 0–4) from the competition page.
2. Put the images in a folder (e.g. `data/images/`). The analysis works with any subset: it keeps the rows of `train.csv`
   whose image is present. My run used **600 images** (322 grade 0, 49 grade 1, 151 grade 2, 28 grade 3, 50 grade 4).
3. `retina-extract --csv data/train.csv --images data/images` writes `results/features.csv`.
