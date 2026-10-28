# AgroDry (Stubby) - complete first-party source snapshot

Prepared for repository review, not published. Original-layout source from `AI Moisture Detection` is retained under `desktop/`.

## Contents

All 15 previously reviewed first-party source/document files are included unchanged, plus the original backend dependency file `Backend/requirments.txt` (original spelling retained). The 16-input-file manifest records source-relative paths, SHA-256, byte size, Python syntax checks and portability notes. The package contains 12 Python sources, two HTML interfaces, the empty original backend README and its dependency file. Data-generation/cleaning code is included; image datasets, label/split CSV datasets, weights, generated videos and caches are excluded.

## Components and variants

- `Backend/`: FastAPI service, inference wrapper, response schema and image-quality helpers.
- `Frontend/`: upload interfaces. Both HTML variants remain separate.
- `afsccnn/`: AFSC training and visualization code; not automatically the deployed model export.
- `Draft1/`: earlier training/finetuning experiments; preserve checkpoint lineage separately.
- `Data/`: CSV preprocessing and train/validation/test splitting scripts, without redistributing underlying datasets.

## Running and model requirements

The dependency file belongs to the backend, not every training script; it lists optional torch/ONNX choices but no trained model. Review its platform-specific comments and install an appropriate runtime only in an isolated environment. From the copied backend directory, the source is designed for an ASGI server such as `uvicorn app:app`; that command has not been runtime-tested in this preparation.

The default execution mode is `stub`, which produces illustrative random moisture independent of the photograph. `QFM_BACKEND`, `QFM_MODEL_PATH`, `QFM_DEVICE` and `QFM_VERSION` select a real export. A requested backend can fall through to a stub when unavailable while metadata still reflects the configured mode. Fixed 0.90 confidence is not calibrated uncertainty. The output-unit heuristic is ambiguous around raw values near 1.2; define export units before claiming real-model performance.

Acquire the original labeled dataset and units, split manifest, exact checkpoint, native trainer, exported TorchScript/ONNX model, transformation contract and held-out test report. No trained accuracy or deployment result is claimed. Confirm the lower-case static frontend reference versus actual `Frontend` directory before Linux deployment. Empty original backend README remains as evidence; this README supplies the release overview.

## Review limits

All 12 Python files parse successfully. No training, model loading, public server or prediction was executed. Credential-shaped literals and private-key headers were scanned without revealing values; no selected file was withheld. Source still contains local/Colab paths and environment configuration, listed per file. Import inventory is an AST inventory, not a resolved dependency lockfile. A public license and contribution attribution have not been inferred or added. Originals are untouched.


## Browse source

[Every published file](FILE-INDEX.md) · [Snapshot hashes](publication-manifest.json)
