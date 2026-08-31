# FactoryVision Model Card

## Model summary

| Field | Description |
| --- | --- |
| Model name | `factoryvision-segmentation` |
| Current registry alias | `candidate` in the local MLflow Model Registry |
| Task | Binary semantic segmentation of manufacturing surface defects |
| Architecture | PyTorch U-Net with 3 input channels, 1 output channel, and 32 base channels |
| Serving format | ONNX Runtime through the FastAPI service |
| Input tensor | `float32`, shape `1 x 3 x 256 x 640`, RGB values normalized to `[0, 1]` |
| Output tensor | `float32` logits, shape `1 x 1 x 256 x 640` |
| Decision rule | Apply sigmoid, then classify pixels as defect when probability `>= 0.5` |

The model produces a pixel-level mask. FactoryVision derives image-level
outputs from that mask: `has_defect`, the maximum defect probability, defect
area fraction, and a bounding box around predicted defect pixels.

## Intended use

The model is intended for learning and prototyping an industrial visual-
inspection workflow. Appropriate uses include:

- demonstrating binary defect segmentation on KolektorSDD2-like images;
- comparing training configurations and inference runtimes;
- supporting an operator with a candidate defect mask and location; and
- exercising a reproducible ML service with tracking, persistence, and
  monitoring.

The output should be treated as decision support. A qualified human or a
validated downstream inspection process must make the final manufacturing
decision.

## Out-of-scope use

Do not use this model as the sole basis for safety-critical decisions,
automatic product rejection, warranty decisions, or claims about defect types
that were not represented in training. It is not a general-purpose visual
inspection model and has not been validated across different cameras,
lighting conditions, materials, production lines, or defect taxonomies.

## Training data and procedure

Training uses the [KolektorSDD2 dataset](https://www.vicos.si/resources/kolektorsdd2/)
and the reusable manifest at `data/processed/splits.csv`. The official test
set is preserved. The official training set is stratified into project train
and validation subsets at an 80/20 ratio.

The baseline configuration is stored in
[`conf/base/parameters.yml`](conf/base/parameters.yml):

- Python 3.11 and PyTorch;
- Adam optimizer with learning rate `0.001` and weight decay `0.0001`;
- a weighted combination of BCE-with-logits loss and soft Dice loss, with
  weights `0.25` and `0.75`;
- a bounded positive-pixel weight to address foreground/background imbalance;
- five epochs by default, batch size `2`, and seed `42`;
- deterministic settings where supported; and
- training-only horizontal flips with probability `0.5`.

The model registry candidate was selected from a controlled mini-study by
final validation IoU, with defect-level F1 as a tie-breaker. The comparison
runs used two epochs to keep the study small and interpretable.

## Evaluation results

The selected baseline produced the following validation results in the
mini-study:

| Metric | Result |
| --- | ---: |
| Pixel IoU | `0.0489` |
| Pixel Dice | `0.0932` |
| Pixel precision | `0.0528` |
| Pixel recall | `0.3998` |
| Defect-level F1 | `0.5029` |
| Validation loss | `0.7560` |

These results are useful evidence that the pipeline runs and that the model
can be compared, but they are not a production-accuracy claim. The numbers
come from a deliberately short two-epoch comparison and the official test
set was not used to select the candidate.

## Runtime validation

The exported ONNX graph was compared with the registered PyTorch model using
the same input. The maximum absolute output difference was `2.29e-05`.

The initial single-thread CPU benchmark reported:

| Runtime | Mean latency | p95 latency | Throughput |
| --- | ---: | ---: | ---: |
| PyTorch via MLflow | `830.17 ms` | `842.97 ms` | `1.205 images/s` |
| ONNX Runtime | `669.14 ms` | `683.99 ms` | `1.494 images/s` |

These measurements are hardware- and configuration-specific. They should be
repeated on the target deployment machine before capacity decisions are made.

## Known limitations and failure modes

- **Low pixel overlap:** the low IoU and Dice scores show that the baseline
  does not yet localize defect boundaries reliably.
- **Class imbalance:** most pixels and images are background/non-defect. The
  weighted loss and sampler reduce this pressure but do not eliminate it.
- **Small or low-contrast defects:** defects that occupy few pixels or resemble
  normal surface texture may be missed.
- **False positives:** reflections, texture, acquisition artifacts, and
  letterboxed regions can be classified as defects.
- **Threshold sensitivity:** changing the `0.5` probability threshold changes
  the precision/recall trade-off and the derived bounding box.
- **Domain shift:** performance may degrade with another camera, material,
  resolution, illumination setup, or production line.
- **Binary scope:** the model predicts defect versus background, not the
  defect category or severity.
- **No calibrated uncertainty:** the reported probability is a model score,
  not a validated probability of failure.
- **Limited validation:** the current evidence is a small study and local
  integration testing, not a statistically powered production acceptance
  test.

## Monitoring and safe deployment

The API exposes health, model metadata, prediction, and Prometheus metrics.
Deployments should monitor error rate, latency, predicted defect rate, and
changes in input data. A sudden metric change should trigger investigation,
not an automatic change to production decisions. The rollback procedure is
documented in [`docs/rollback.md`](docs/rollback.md).

## Reproducibility

Training, MLflow tracking, ONNX export, and serving commands are documented in
the [README](README.md). The trained weights and generated ONNX file remain
outside Git as generated artifacts. Reproducible model identity therefore
requires recording the Git commit, MLflow run ID, model alias/version, dataset
manifest version, configuration, and device used.

## License and attribution

This card describes the FactoryVision model code and its evaluation. The
underlying dataset has its own terms; see the [Data Card](DATA_CARD.md) and the
official [KolektorSDD2 resource page](https://www.vicos.si/resources/kolektorsdd2/).
