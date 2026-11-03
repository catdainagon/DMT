# DMT
Dual-endpoint Multiscale unbalanced Transport for anti-money-laundering detection

This repository contains a Python research implementation of dual-endpoint multiscale transport (DMT) for transaction-level anti-money-laundering detection. The project studies how prior fund movements at a payment's sender and receiver can complement tabular and temporal representations. It includes data preparation, model training, benchmark comparisons, parameter search, calibration and analysis of allocation-derived concepts.

The associated manuscript, **“Dual-endpoint multiscale unbalanced transport for anti-money-laundering detection,”** has been submitted to **Expert Systems with Applications (ESWA)** and is currently under review. The full code will be made available after the review process is complete.

## Model and experiments

DMT combines four prediction paths:

- **Tabular:** piecewise-linear embeddings of transaction attributes and historical summaries.
- **Temporal:** separate sender and receiver encoders, using either GRUs or causal Transformers with query-based attention.
- **Transport:** learned allocations between earlier inflows and outflows at both endpoints, evaluated over 6-, 24-, 168- and 720-hour horizons.
- **Concept:** twelve allocation-derived quantities describing funding coverage, residual funding, turnover and fragmentation.

A learned gate combines the expert logits. The sender allocation includes the queried payment as outgoing demand; the receiver allocation summarizes preceding transactions.

The full experiment suite evaluates DMT, a variant with transport and concepts removed, a deterministic-proxy variant, and 19 comparator families:

| Group | Comparators |
|---|---|
| Tabular | Raw logistic regression, raw XGBoost, history-informed XGBoost and LightGBM, historical MLP |
| Graph and allocation | GraphSAGE, GAT, TGAT-style, TGN-style, MultiGIN, line graph, transport-concept classifier |
| Snapshot | EvolveGCN–GRU and six combinations of GCN or GAT with GRU, LSTM or Transformer heads |

Graph comparators use local retained-history adaptations. The suite also studies history length, transport parameters, population construction, concept behavior and numerical convergence. Evaluation includes average precision, calibration, alert-budget performance, seed variation and paired bootstrap intervals.

## Installation

Use Python 3.11 or later. Native ARM64 Python 3.12 is the tested configuration on Apple Silicon. From the directory containing `setup.sh` and `run.sh`:

```bash
bash setup.sh
./run.sh doctor
```

Setup creates an isolated `.venv`, installs `requirements-tested.txt`, checks dependencies and runs the tests. `requirements.txt` contains broader dependency bounds; `requirements-lock-macos-arm64-py312.txt` records the resolved packages for the tested macOS environment. Use `AML_PYTHON` to select a Python interpreter or `bash setup.sh --recreate` to rebuild the environment while preserving the previous installation.

AMLSim generation additionally requires a Java 8 JDK, Maven, Git and MASON 20. These system dependencies are checked separately from Python package installation.

## Running experiments

Inspect the experiment plan before starting a benchmark:

```bash
./run.sh --suite full --dry-run
```

Run the complete suite across all four datasets:

```bash
./run.sh --suite full --run-name full_benchmark
```

The workflow acquires missing datasets, prepares features and histories, selects configurations, trains final models, fits calibration and exports analyses. The default grid schedules 1,308 training fits, including five final seeds and the sensitivity studies.

For a small development check on IBM HI-Small:

```bash
./run.sh --suite full --datasets ibm_hi_small \
  --development --max-targets 8000 --max-epochs 1 --seeds 11 \
  --stages primary diagnostics analysis --run-name development_check
```

Development mode uses a label-enriched target subset and marks its outputs `pipeline_validation_only`. These outputs check the workflow; they do not estimate performance at the dataset's natural prevalence. Omit `--development` for benchmark runs.

Use `--datasets` to select datasets and `--stages` to select work:

| Stage | Work performed |
|---|---|
| `primary` | Configuration selection and final model fits |
| `purge` | Comparisons of target and historical-context exclusion policies |
| `sensitivity` | DMT history and transport parameter studies |
| `diagnostics` | Allocation examples, concept agreement and solver accuracy |
| `analysis` | Ensembles, uncertainty, calibration and subgroup summaries |

Later stages use the selected configurations and model outputs from `primary`. Repeating a command reuses completed jobs when the recorded code, settings and inputs match. Keep the dependency environment fixed when resuming, and use a new `--run-name` after changing the experiment configuration.

## Parameter search

The default uses a compact grid. Bounded random search explores learning rates, network widths, regularization and model-specific settings. Candidate configurations are ranked by selection-partition average precision, with Brier score as the tie-breaker. Calibration and test partitions remain separate from parameter selection.

```bash
# Search twenty candidates per model and save the selected configurations
./run.sh --suite full --search random --search-trials 20 \
  --search-only --run-name full_random20

# Continue with final fits and analyses using the saved selections
./run.sh --suite full --search random --search-trials 20 \
  --run-name full_random20
```

Search ranges are defined in `configs/search_space.json`. Pass a modified copy with `--search-space`. `--search-seed` controls parameter sampling, while `--search-seeds 11 22 33` repeats each candidate across training seeds. Final evaluation defaults to seeds 11, 22, 33, 44 and 55. Use `--batch-size`, `--patience` and `--max-epochs` to set training budgets, and add `--dry-run` to inspect the resulting fit count. Development mode caps the search at one candidate per model.

## Data and outputs

The data adapters support TransXion, AMLNet-v2, IBM HI-Small and AMLSim. Public-source downloads are versioned or checksum-verified. AMLSim populations are newly generated from the profiles in `configs/amlsim/`, with their realized counts and generator provenance recorded.

By default, `Data/` and `Experiments/` are created alongside the code directory:

```text
AML/
├── Codes/              # Source code and configuration
├── Data/
│   ├── raw/            # Downloaded source files
│   ├── canonical/      # Canonical transaction tables
│   ├── processed/      # Splits, features, histories and proxies
│   └── generator/      # AMLSim source, environment and generated records
└── Experiments/        # Named runs, checkpoints, predictions and analyses
```

`--root` selects a different parent directory for `Data/` and `Experiments/`. `--no-download` requires the canonical datasets to be available locally. The source tree contains the implementation in `aml_pipeline/`, configuration files in `configs/`, validation tests in `tests/` and supporting checks in `scripts/`.

Histories contain strictly earlier transactions, and preprocessing statistics are fitted on training data. The default preparation uses same-currency histories and excludes crossing positive groups from prediction targets while retaining their earlier transactions as observable context. Alternative population policies are evaluated by the `purge` stage. Run records include source hashes, split membership, model settings, selected configurations, calibration parameters and aligned predictions.

## Hardware and validation

`--device auto` selects an available accelerator for neural models, including MPS on Apple Silicon and CUDA on supported hosts. Use `--device cpu`, `--device mps` or `--device cuda` to choose explicitly. Boosting, logistic regression, data preparation and statistical analysis run on the CPU.

MPS uses float32 model arithmetic. Timestamp ordering, horizon membership and snapshot boundaries are computed from integer microseconds before transfer. Performance depends on the model and batch size; device selection and runtime measurements are recorded per fit.

```bash
./run.sh doctor
./run.sh test -q
```

The tests cover temporal information boundaries, data alignment, model computations, transport numerics, calibration and experiment resumption. Native MPS checks run when Metal is available. Numerical and development checks are recorded separately from benchmark results.
