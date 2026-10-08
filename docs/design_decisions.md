# Design decisions

Why the pipeline is built the way it is. Sources are in [references.md](references.md); where there is **no** literature behind a choice, I say so.

## 1. Find the retina instead of assuming a centred circle
APTOS images come from different cameras: some are cut at the top and bottom, some have wide black borders, and the retina is not always centred. A fixed circle either cuts real retina or keeps black corners. I locate the retina from the image itself (not-black pixels, largest blob), crop to it, pad to a square and resize to 512 px. Everything that follows is measured **inside that mask**.

## 2. Measure per unit of retina, not in raw pixels
Vessel area, skeleton length, junction and lesion counts are expressed as densities (per retina pixel or per 1,000 skeleton pixels). Otherwise an image with more black border, or a different size, looks like a different eye.

## 3. Tortuosity per branch, with the real path length
For every vessel branch (the skeleton between two junctions), tortuosity is arc length divided by the straight distance between its ends. The arc length is the geodesic path along the branch with diagonal steps counting √2, so a perfectly straight diagonal vessel scores exactly 1 (checked in the tests). Junctions are counted as clusters, not as pixels, so one bifurcation counts once.
*Limit:* on thin pixel skeletons, staircase effects add a little length to wavy vessels; the values are for comparing images, not absolute physical tortuosity.

## 4. Lesions: thresholds well above the noise, and vessels masked out
A top-hat filter responds to every small bright or dark structure. Thresholding it at the noise level (for example with Otsu) "finds" hundreds of lesions in every image, healthy or not. I threshold at the median plus six robust standard deviations of the response inside the retina, with a minimum contrast. For dark lesions I first remove the vessels (they are dark too).
*Limit:* these are simple heuristics of my own, not validated against lesion annotations; the optic disc and vessel reflections can still be picked up as bright spots.

## 5. Always compare with an "image quality only" model, and report a within-camera AUC
APTOS images differ by camera, brightness, sharpness and size, and in my 600 photos the image size alone nearly gives away the diagnosis (for example, none of the 61 photos of 2048 × 1536 has DR, and 97 % of those of 3216 × 2136 do). A model can score well by learning the camera instead of the eye. I fit the same models on quality features only and report them next to the morphology models. I also report a **within-size AUC**: the AUC is computed among images of the same size (a proxy for the camera) and averaged with weights proportional to the number of healthy-DR pairs in each size. It uses only sizes with at least five healthy and five DR images, so it covers 351 of the 600 images for the any-DR question.
*Limit:* image size is a proxy for the camera, not the camera itself; and the model is still trained on all images together, so SHAP and the pooled numbers contain camera effects (for example, mean vessel width goes one way pooled and the other way within sizes).

## 6. Cross-validation on every labelled image, not one 75/25 split
With a few hundred images, one test split gives a wide, lucky-or-unlucky number. I use repeated stratified 5-fold cross-validation (3 repeats), so every image is tested exactly once per repeat and no model sees its own test images. Folds are stratified on label × image size, and near-duplicate photos are grouped (a 256-bit hash of the vessel pattern; a first attempt with a hash of the raw image was useless, because fundus photos all look alike at low resolution) so the same eye never lands on both sides of a split. I found two duplicated pairs, one with conflicting grades.

## 7. The decision threshold is chosen without looking at the test fold
For screening, I want to catch at least 90 % of the cases. The threshold that achieves this is picked from an inner cross-validation **inside each training fold**, then applied to the held-out fold. (Choosing it on the same images it is later reported on makes sensitivity look better than it is.) Sensitivity and specificity come with Wilson 95 % intervals; AUC comes with a bootstrap interval over images.
*Why 90 %:* screening systems are judged at high-sensitivity operating points (Gulshan et al. 2016; Abràmoff et al. 2018).

## 8. Two binary questions, and the five grades reported separately
*Any DR* (grade ≥ 1) and *referable DR* (grade ≥ 2, moderate or worse; Gulshan et al. 2016) are the clinically meaningful screening questions. The five-grade task is reported with accuracy, macro-F1 and quadratic weighted kappa **next to a "always predict the most common grade" baseline**, because with 28 severe and 50 proliferative cases among 600 images, accuracy alone is misleading.

## 9. A label-permutation test
To know what "no signal" looks like for this data size, I re-run the whole cross-validation 50 times with shuffled labels. The real AUC is compared with that null distribution; for the within-size AUC the labels are shuffled only among images of the same size.

## 10. Honest use of a third-party network
The vessel network (FCU-Net, MIT licence, vendored with attribution) was trained on DRIVE, STARE and CHASE, not on APTOS. APTOS has no vessel annotations, so **I make no claim about segmentation accuracy here**; I only check the masks by eye on examples. The weights are downloaded on first use and verified by SHA-256; they are not stored in the repository.

## Not done (and why it matters)
- **External validation** on other datasets (EyePACS, Messidor-2, IDRiD): the main missing piece, since APTOS cameras differ from others.
- **The full APTOS training set** (3,662 images): I only had 600 locally.
- **Macular edema**, which the APTOS labels do not cover.
- **A deep-learning baseline** trained end to end, to quantify what the interpretable route gives up.
- **Calibration** of the predicted probabilities.
