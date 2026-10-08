# References

Sources I rely on, with what each one supports here (and what it does not). Items 1–3 were checked in PubMed.

| # | Reference | What it supports here |
|---|---|---|
| 1 | Wilkinson CP, Ferris FL, Klein RE, et al. **Proposed international clinical diabetic retinopathy and diabetic macular edema disease severity scales.** *Ophthalmology* 2003;110:1677–1682. [doi:10.1016/S0161-6420(03)00475-5](https://doi.org/10.1016/S0161-6420(03)00475-5) | The five-stage severity scale behind the 0–4 labels (no DR, mild, moderate, severe, proliferative): three low-risk stages, then severe non-proliferative and proliferative disease. |
| 2 | Gulshan V, Peng L, Coram M, et al. **Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs.** *JAMA* 2016;316:2402–2410. [doi:10.1001/jama.2016.17216](https://doi.org/10.1001/jama.2016.17216) | "Referable" DR is defined as moderate or worse (the task I call *referable DR*, grade ≥ 2), and screening algorithms are reported at an operating point chosen on development data for high sensitivity. This is a deep network trained on 128,175 images; it is context for the task, not a benchmark for a 600-image hand-crafted model. |
| 3 | Abràmoff MD, Lavin PT, Birch M, Shah N, Folk JC. **Pivotal trial of an autonomous AI-based diagnostic system for detection of diabetic retinopathy in primary care offices.** *NPJ Digit Med* 2018;1:39. [doi:10.1038/s41746-018-0040-6](https://doi.org/10.1038/s41746-018-0040-6) | A prospective trial of an approved screening system, with pre-specified targets of about 85 % sensitivity and 82.5 % specificity for more-than-mild DR. It shows why the screening question is asked at a fixed high-sensitivity operating point and not at accuracy. |
| 4 | Gurrola-Ramos J, Dalmau O, Alarcón TE. **Factorized U-net for retinal vessel segmentation.** *Mexican Conference on Pattern Recognition* (Springer), 2022, pp. 181–190. Code and weights: [JavierGurrola/FCU-Net](https://github.com/JavierGurrola/FCU-Net) (MIT) | The vessel-segmentation network used here, with its public weights trained on DRIVE, STARE and CHASE. *(Not indexed in PubMed; reference details taken from the authors' repository.)* |

## What the literature does *not* say

- None of these sources says that hand-crafted vessel measurements should reach any particular accuracy on APTOS; the numbers I report are my own, on one small subset.
- The lesion detectors (top-hat filters with a robust threshold) are simple heuristics of my own; I did not validate them against lesion annotations.
- The APTOS images come from several cameras and clinics. I did not find a source that quantifies this for the subset used here, which is why I measure it directly (the image-quality-only baseline).
