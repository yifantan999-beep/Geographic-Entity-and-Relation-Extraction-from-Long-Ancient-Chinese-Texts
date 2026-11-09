# LongGeoIE

LongGeoIE is a two-stage information extraction system for geographic entities and relations in ancient Chinese text. It combines span-based entity recognition with document-level relation extraction to handle dense entity candidates and relations whose evidence is spread across long contexts.

The project is built on the ITER implementation and extends it with boundary/type-aware NER, tokenizer-aware overlapping windows, global-offset entity merging, and a document-level relation model with local/global evidence fusion.

## Current Main Model

The reported IND model uses a Chinese BERT-base encoder initialized from google-bert/bert-base-chinese, then fine-tuned on the IND training split.

**Stage 1: entity recognition**

- ITER-based span/entity extraction
- TypeAux and start/end BoundaryAux losses
- Span-Biaffine span scoring
- Tokenizer-aware windowing with global offsets
- Fixed-threshold decoding followed by NMS

**Stage 2: document-level relation extraction**

- Relation scoring over predicted document entities
- Learned local/global evidence fusion
- Distance-conditioned fusion gate and long-distance head
- Distance experts, direction and window-position features
- Pair-type compatibility constraints

The IND v8 fixed inference setup uses NER span/type thresholds 0.80/0.50, NMS, and relation threshold 0.24. The optional IND Pair-Link v2 variant is a separate experiment and is not the LongGeoIE row in the efficiency benchmark.

## Results

### Original IND Test Set

The original IND test split contains 467 documents. The fixed v8 LongGeoIE result is:

| Task/metric | Precision | Recall | F1 |
|---|---:|---:|---:|
| Strict NER | 78.47 | 79.61 | 79.04 |
| Boundary NER | 85.85 | 87.09 | 86.47 |
| Strict end-to-end RE | 44.94 | 64.39 | 52.93 |
| RE on matched entities | 56.79 | 81.03 | 66.78 |

These scores are percentages. Strict NER requires exact span boundaries and entity type. Strict end-to-end RE additionally requires the relation label, typed endpoints, and direction to be correct.

### Long-Context IND-LC Test Set

The project also contains a constructed long-context benchmark derived from IND. Its documents concatenate complete source samples while preserving source annotations and provenance. The v8 fixed diagnostic-easy test result recorded in the project is Strict NER F1 75.09, Boundary NER F1 82.46, and Strict end-to-end RE F1 62.27.

The diagnostic-easy split was used during targeted model development. Treat it as a diagnostic result, not as an untouched independent test set. See docs/PRIMARY_EXPERIMENT.md and docs/FROZEN_HOLDOUT_V6_ZH.md for split-specific scope and caveats.

### Parameter Count and IND Inference Time

Measured on the complete 467-document IND test split:

| Stage 1 params | Stage 2 params | Total params | Inference time | Time/document | Throughput |
|---:|---:|---:|---:|---:|---:|
| 153.604M | 12.707M | 166.311M | 26.511 ± 0.006 s | 56.769 ms | 17.615 docs/s |

The measurement used an NVIDIA RTX 4080, batch size 2, one warm-up run, and three measured runs. It includes Stage 1 inference/decoding, document-level NER merging/NMS, and Stage 2 relation extraction. Checkpoint loading, dataset preparation, metric calculation, and output serialization are excluded. Baseline timing results were collected on different hardware/settings, so the project's baseline latency table is descriptive rather than a controlled speed ranking.

Detailed benchmark files, when present:

~~~text
artifacts/analysis/longgeoie_ind_efficiency/benchmark.json
artifacts/analysis/longgeoie_ind_efficiency/benchmark_runs.csv
artifacts/analysis/longgeoie_ind_efficiency/benchmark_table.md
artifacts/analysis/longgeoie_ind_efficiency/benchmark_table.tex
~~~

## Repository Layout

~~~text
cfg/wenyanwen/       Dataset and model configurations
datasets/            IND and derived long-context data (not included in source control)
iter/                 ITER model, windowing, losses, metrics, and relation modules
models/               Trained checkpoints (not included in source control)
scripts/              Training, evaluation, analysis, and benchmark entry points
artifacts/            Predictions, metrics, calibration, and analysis outputs
docs/                 Experiment protocols, project guides, and paper materials
tests/                Unit and pipeline tests

train_long_context.py       Stage 1 long-context training entry point
train_document_relation.py  Stage 2 document relation training entry point
infer_long_context.py       Window inference, document merging, and evaluation
evaluate_long_context.py    Recompute analysis metrics from saved predictions
~~~

Important implementation files:

- iter/modeling_iter2.py: active ITER model implementation, including NER auxiliary heads/losses.
- iter/datasets/windowing.py: tokenizer-aware overlapping-window construction and coverage metadata.
- iter/modules/document_relation.py: document-level relation fusion and distance-conditioned modules.
- iter/misc/document_metrics.py: global-offset strict NER/RE and long-context metrics.

Multiscale geographic representations and fact-decomposition components also exist in the repository, but they are separate experimental branches and are not the current v8 fixed LongGeoIE configuration.

## Environment

Python 3.10 or newer is required. Install the project dependencies in a virtual environment:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
~~~

For CUDA training, install a PyTorch build compatible with the host's CUDA driver before installing the remaining dependencies if needed. For offline use, place the bert-base-chinese model/tokenizer under the configured local path and set OFFLINE=1.

The usual local pretrained-model path used by the scripts is:

~~~text
local-bert/google-bert/bert-base-chinese
~~~

The trained IND checkpoints are expected at:

~~~text
models/wenyanwen_ind_v8_fixed/bert-base-chinese/run_01/
models/wenyanwen_ind_v8_fixed/bert-base-chinese/run_01/document_relation_distance_experts_v8/
~~~

The checkpoint files are intentionally not distributed with the source tree. Copy or restore the trained weights before running inference.

## Reproduce the IND v8 Fixed Experiment

Run the full Stage 1 training/calibration, Stage 2 training/calibration, and final test workflow on a Linux server with the IND data and pretrained model available:

~~~bash
cd ~/codex_project
export DEVICE=cuda:0
export OFFLINE=1
bash scripts/run_ind_v8_fixed_all.sh
~~~

The stages can also be run independently:

~~~bash
bash scripts/run_ind_v8_fixed_stage1.sh
bash scripts/run_ind_v8_fixed_stage2.sh
bash scripts/run_ind_v8_fixed_final_test.sh
~~~

For the fixed-threshold IND test specified by the v8 fixed protocol, use:

~~~bash
DEVICE=cuda:0 OFFLINE=1 \
  bash scripts/run_ind_v8_fixed_fixed_test.sh
~~~

That script uses NER span threshold 0.80, type threshold 0.50, NMS, and relation threshold 0.24; it refuses to overwrite an existing result directory. Set OUTPUT_DIR to a new path for another run.

The main artifacts are:

~~~text
models/wenyanwen_ind_v8_fixed/bert-base-chinese/run_01/
models/wenyanwen_ind_v8_fixed/bert-base-chinese/run_01/document_relation_distance_experts_v8/
artifacts/inference/ind_v8_fixed_fixed_test/test_metrics.json
artifacts/inference/ind_v8_fixed_fixed_test/test_predictions.jsonl
~~~

## Reproduce the IND Efficiency Measurement

With the same checkpoints restored, run:

~~~bash
DEVICE=cuda:0 OFFLINE=1 \
  WARMUP_RUNS=1 REPEATS=3 BATCH_SIZE=2 \
  bash scripts/run_longgeoie_ind_efficiency.sh
~~~

This generates machine-readable raw measurements and paper tables under:

~~~text
artifacts/analysis/longgeoie_ind_efficiency/
~~~

## Long-Context Experiments

The long-context pipeline uses the same document-level inference/evaluation entry point. The exact dataset config and checkpoint must match; do not silently substitute the IND checkpoint or a different split.

An example diagnostic-easy fixed test command is:

~~~bash
DEVICE=cuda:0 OFFLINE=1 \
  bash scripts/run_primary_diagnostic_easy_fixed_test.sh
~~~

This diagnostic-easy split has been used for targeted development and must not be presented as an untouched holdout. Consult docs/PRIMARY_EXPERIMENT.md for its limitations and the corresponding configuration.

For windowing, training, metric definitions, distance buckets, and output interpretation, see:

- docs/LONG_CONTEXT_TRAIN_INFERENCE_GUIDE.md
- docs/LONG_CONTEXT_STAGE1_PROTOCOL.md
- docs/experiment4_5_guide_zh.md

## Data and Output Notes

The IND source files are expected under datasets/ind/:

~~~text
datasets/ind/train.json
datasets/ind/dev.json
datasets/ind/test.json
datasets/ind/types.json
~~~

The repository's ignore rules exclude datasets/, models/, generated safetensors caches, and most bulky evaluation artifacts. These files must be provisioned separately and should follow the documented split definitions. Do not use test labels for threshold selection; calibration and model selection belong on dev/eval.

infer_long_context.py writes predictions, errors, metrics, and comparison tables to the selected output directory. The primary metrics include:

- ner_strict: exact global span and type match.
- ner_boundary: exact global span match, ignoring entity type.
- re_strict_end_to_end: strict relation match including typed endpoints and direction.
- re_on_matched_entities: relation evaluation conditional on both endpoint boundaries being detected.
- length_buckets, relation_distance_buckets, and window_relation_coverage: long-context diagnostics.

For Oracle and error-propagation analysis, as well as length/distance/cross-window summaries, see docs/experiment4_5_guide_zh.md and run scripts/run_experiment4_5.sh on the saved prediction JSONL.

## Tests

Run the repository tests with:

~~~bash
python -m pytest -q
~~~

The tests cover configuration, window construction, Stage 1 inference behavior, document-level relation modules, and analysis utilities. GPU training and the full IND benchmark require the separately provisioned model weights and data.

## Citation

The base ITER implementation is described in the EMNLP 2024 paper:

~~~bibtex
@inproceedings{iter2024,
  title     = {ITER: Iterative Transformer-based Entity Recognition and Relation Extraction},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2024},
  year      = {2024},
  url       = {https://aclanthology.org/2024.findings-emnlp.655/}
}
~~~

Please also cite the original IND dataset and any baseline methods whose results are used in comparisons.
