# Measuring Differences between Conditional Distributions using Kernel Embeddings

Research code for comparing conditional distributions using conditional maximum mean discrepancy (CMMD), including tests with equal or unequal covariate marginals, different CMMD levels, and doubly robust estimators.

Core statistics and tests are in `src/cmmd.py`, cross-fitted doubly robust estimators in `src/dr_cmmd.py`, and kernels and data generators in the other `src/` modules. Experiment scripts are in `experiments/`.

## Setup

Use Python 3.12 (tested with Python 3.12.1 on macOS). From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate.bat` (Command Prompt) or `.venv\Scripts\Activate.ps1` (PowerShell).

## Run experiments

Start with a quick synthetic-data plot:

```bash
python experiments/synthetic_plot_data.py
```

This saves `figs/synthetic/data_plot_diff_marginal.pdf`. Run any script below in the same way.

| Script in `experiments/` | Purpose |
| --- | --- |
| `synthetic_plot_data.py`, `synthetic_plot_test_stat.py` | Synthetic samples and statistic distributions |
| `synthetic_plot_error_rate.py` | Type I error and power versus sample size |
| `synthetic_power_plot.py`, `synthetic_dimension.py` | Power versus covariate mean and dimension, including sample-split CMMD level selection |
| `synthetic_mixture_rho.py`, `synthetic_mixture_prop.py` | Power versus reference mixture weight and sample allocation |
| `lvls_cmmd.py` | Power comparisons across CMMD levels |
| `dr_cme.py`, `dr_cmmd.py` | Standard and doubly robust embedding/statistic estimates |
| `dr_cmmd_error.py`, `dr_cmmd_rejection_rate.py` | Estimation error and rejection rates for doubly robust CMMD |
| `mnist_error_rate.py` | Type I error and power on MNIST under covariate shift |

Settings are defined inside each script, usually in the final `__main__` block; there are no command-line options. Adjust sample sizes, `n_trials` (repeated trials), and `B` (resamples) for shorter runs. In error-rate scripts, `error="type1"` measures Type I error and `error="type2"` measures power. Change `setting` where available to run the other scenarios.

Scripts save PDFs under `figs/{synthetic,lvls,dr,mnist}/` and print numerical summaries to the terminal. 

## MNIST data

Download and decompress the MNIST test files [t10k-images-idx3-ubyte.gz](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz) and [t10k-labels-idx1-ubyte.gz](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz) into `data/raw_mnist/`, removing the `.gz` extension (data are not included in the repository), then run:

```bash
python src/clean_mnist.py
python experiments/mnist_error_rate.py
```

Preprocessing normalizes the images and reduces them to 100 PCA components, saving `data/clean_mnist/mnist_test.csv`. The experiment uses digit labels as covariates and these components as outcomes.
