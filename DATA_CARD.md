# FactoryVision Data Card

## Dataset summary

FactoryVision uses the [Kolektor Surface-Defect Dataset 2
(KolektorSDD2)](https://www.vicos.si/resources/kolektorsdd2/), provided by
Kolektor Group and annotated in a controlled industrial environment. It is a
binary surface-defect segmentation dataset: each image is paired with a
pixel-level ground-truth mask.

The official source describes approximately 230 x 630 pixel, three-channel
images containing several surface-defect types such as scratches, minor
spots, and other surface imperfections.

## Composition

The local exploratory analysis found 3,335 usable image/mask pairs and no
missing masks:

| Class | Images | Masks | Share of images |
| --- | ---: | ---: | ---: |
| Defect | 356 | 356 | 10.7% |
| Non-defect | 2,979 | 2,979 | 89.3% |
| Total | 3,335 | 3,335 | 100.0% |

An image is classified as `defect` when its mask contains at least one
nonzero pixel. A `non-defect` image has an empty mask. The mask is therefore
the ground truth for both the pixel-level segmentation task and the derived
image-level defect label.

## Official and project splits

The official test split is preserved. The official training split is divided
stratifiably into project training and validation subsets at an 80/20 ratio.
The resulting reusable manifest is
[`data/processed/splits.csv`](data/processed/splits.csv).

| Project split | Defect | Non-defect | Total |
| --- | ---: | ---: | ---: |
| Train | 197 | 1,668 | 1,865 |
| Validation | 49 | 417 | 466 |
| Test | 110 | 894 | 1,004 |

The test counts match the official source. The train and validation counts are
the project-specific subdivision of the official training data, not a new
random test set.

## Annotation format

The raw directory contains pairs such as:

```text
data/raw/kolektor-sdd2/train/10021.png
data/raw/kolektor-sdd2/train/10021_GT.png
```

The `_GT.png` file is read as a grayscale mask. During loading, every
nonzero pixel becomes `1.0` and background becomes `0.0`, producing a binary
`float32` tensor with shape `1 x 256 x 640` after preprocessing.

## Preprocessing

`KolektorSDD2Dataset` applies the following deterministic preparation:

1. Read the image and mask with OpenCV.
2. Convert the image from BGR to RGB.
3. Resize the image and mask proportionally into the configured `256 x 640`
   canvas.
4. Use bilinear interpolation for the image and nearest-neighbor
   interpolation for the mask so labels remain discrete.
5. Add symmetric image padding and zero-valued mask padding where required.
6. Convert the image to channel-first `float32` and scale pixels to `[0, 1]`.
7. Convert the mask to binary `float32` values in `{0, 1}`.

Training-only horizontal flips can be enabled in
[`conf/base/parameters.yml`](conf/base/parameters.yml); image and mask are
transformed together. Validation and test samples are not augmented.

## Data quality and limitations

- The dataset is highly imbalanced toward non-defect images, so accuracy alone
  would be misleading. Evaluation uses IoU, Dice, precision, recall, and
  defect-level F1.
- Images were captured in a controlled industrial environment. Results may
  not transfer to other cameras, surfaces, illumination, or production lines.
- The labels identify defect pixels but do not provide a general severity
  scale or a complete defect taxonomy for FactoryVision.
- Small or low-contrast defects can be difficult to annotate consistently and
  difficult for a baseline model to localize.
- The project manifest depends on the DVC-tracked dataset layout. Raw images
  are intentionally not committed to Git.
- The public sample images in `assets/dataset/` are documentation examples,
  not a replacement for restoring the complete DVC dataset.

## Intended and inappropriate use

Appropriate uses include research, education, reproducibility exercises,
segmentation benchmarking, and prototyping an industrial inspection pipeline.
The data and resulting model should not be used as the sole basis for
safety-critical decisions or production quality release without separate
validation on representative line data.

## Provenance, license, and citation

The official resource page states that the images were provided and annotated
by Kolektor Group, and that the dataset is licensed under
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Commercial use requires contacting the dataset authors.

When using the dataset, cite the source paper:

> Božič, Jakob; Tabernik, Domen; Skočaj, Danijel. “Mixed supervision for
> surface-defect detection: from weakly to fully supervised learning.”
> *Computers in Industry*, 2021.

See the official [KolektorSDD2 resource page](https://www.vicos.si/resources/kolektorsdd2/)
for the authors, download location, license, and citation details.

## Maintenance

Any change to the dataset version, split manifest, preprocessing, or label
interpretation should update this card and be recorded alongside the DVC
version and experiment metadata.
