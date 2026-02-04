# neural-ssm

PyTorch implementations of neural state-space models (SSMs), including LRU-based and L2-bounded variants for stable system modeling and system identification.

This repo focuses on:
- fast sequence processing via parallel scan
- controllable/stable recurrent parametrizations
- practical training scripts for benchmark datasets

---

## Highlights

- Multiple recurrent SSM blocks under one API (`DeepSSM`)
- L2-bounded variants with gain control (`gamma`) and certificate-oriented scaling
- Loop and scan execution modes (`mode="loop"` or `mode="scan"`)
- Optional Lipschitz feedforward layers (`LMLP`, `LGLU`, `TLIP`)
- End-to-end benchmark training script with Optuna hyperparameter search

---

## Installation

### 1) Clone the repository

```bash
git clone https://github.com/LeoMassai/neural-ssm.git
cd neural-ssm
```

### 2) Create and activate an environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
```

### 3) Install the package

```bash
pip install -e .
```

Optional development dependencies:

```bash
pip install -e ".[dev]"
```

Optional example dependencies:

```bash
pip install -e ".[examples]"
```

For benchmark and control scripts in `Test_files`, you may also need:

```bash
pip install scipy optuna nonlinear-benchmarks control
```

---

## Quick Start

```python
import torch
from neural_ssm import DeepSSM, SSMConfig

cfg = SSMConfig(
    d_model=8,
    d_state=8,
    n_layers=3,
    ff="LGLU",      # GLU | MLP | LMLP | LGLU | TLIP
    param="l2n",    # lru | l2ru | zak | l2n | l2nt | tv
    gamma=2.0,
    train_gamma=True,
)

model = DeepSSM(d_input=1, d_output=1, config=cfg)

u = torch.randn(16, 300, 1)            # (batch, time, input_dim)
y, state = model(u, mode="scan")       # or mode="loop"
print(y.shape)                          # (16, 300, 1)
```

---

## Core API

Main classes live in `src/neural_ssm/ssm/lru.py` and are re-exported at package level:

- `DeepSSM`: encoder -> stacked SSM blocks -> decoder
- `SSMConfig`: dataclass for architecture and parametrization settings
- `LRU`: complex diagonal LRU with scan/loop simulation
- `L2RU`: L2-stable LRU-style block
- `lruz`: alternative LRU parametrization
- `PureLRUR`: recurrent-only wrapper block
- `SimpleRNN`: baseline recurrent model

Feedforward/static layers:

- `GLU`, `MLP` in `src/neural_ssm/static_layers/generic_layers.py`
- `LMLP`, `L2BoundedGLU`, `TLIP` in `src/neural_ssm/static_layers/lipschitz_mlps.py`

Additional modules:

- `REN` model in `src/neural_ssm/rens/ren.py`
- experimental/time-varying SSM variants in `src/neural_ssm/ssm/experimental.py`

---

## Benchmark Training

The main benchmark script is:

```bash
python Test_files/Benchmark.py
```

What it does:
- loads the Wiener-Hammerstein benchmark from `nonlinear_benchmarks`
- trains a `DeepSSM` model
- computes validation RMSE
- saves checkpoints and plots in `checkpoints/`

### Hyperparameter optimization (Optuna)

`Test_files/Benchmark.py` includes an Optuna study that tunes:
- number of SSM layers (`n_layers`)
- learning rate
- gain parameter (`gamma`)

Best parameters are written to:

- `checkpoints/optuna_best_params.json`

You can adjust search settings in the script via `HyperOptConfig`.

---

## Project Structure

```text
src/neural_ssm/
  __init__.py
  ssm/
    lru.py
    scan_utils.py
    mamba.py
    experimental.py
  static_layers/
    generic_layers.py
    lipschitz_mlps.py
  rens/
    ren.py

Test_files/
  Benchmark.py
  Example.py
  Control_Example.py
  ...
```

---

## Development

Run style/type/test tooling (if installed via `.[dev]`):

```bash
ruff check .
black .
mypy src
pytest
```

---

## Citation

If you use this code in research, please cite:

**Free Parametrization of L2-bounded State Space Models**  
https://arxiv.org/abs/2503.23818

---

## Notes

- This is an active research codebase; APIs and scripts may evolve.
- Some scripts in `Test_files/` are exploratory and may require extra dependencies/datasets.
- If your local scripts import via `src.neural_ssm...`, that works from repository root.  
  For installed-package usage, prefer `import neural_ssm`.





