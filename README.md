# tensorpow_benchmark

Reproducible benchmark protocol and results for `tensorpow` (MSc thesis to journal article).

## Environment Details

The benchmarks in this repository were designed to run on the following specific hardware configuration:

- **OS**: Cachy OS
- **CPU**: Intel Core i5 14600KF (running at base clock speeds)
- **RAM**: 32 GB DDR5 6000MHz CL36

## Benchmark Protocol

This benchmark replaces/supplements the development timings inherited from the original MSc thesis. It compares the virtual-block implementation in `tensorpow` with a brute-force Kronecker construction (while memory allows).

The script evaluates two fixed-seed test families:
1. Two-term expression: `0.5 A^{⊗n} - 0.5 B^{⊗n}`
2. Three-term expression: `0.25 A^{⊗n} + 0.25 B^{⊗n} - 0.5 C^{⊗n}`

A, B, and C are independently generated random probability density matrices (created by generating random complex matrices, making them Hermitian, and normalizing their trace to 1).

### Measured Metrics
- **Warm virtual-block time**: `TensorPowerCalculator` is reused.
- **Cold-process virtual-block time**: Each measurement starts a fresh Python process (measuring initialization overhead).
- **Direct Kronecker time**: Full matrix construction + dense SVD.

## Running the Benchmarks

The benchmark script supports continuous saving and can be safely cancelled and resumed. If you stop the script midway, simply re-run the same command, and it will pick up where it left off, skipping already completed test cases and repetitions.

### 1. Setup Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorpow==0.3.0
```

### 2. Pre-Download Representation Data (Recommended)

`tensorpow` downloads precomputed representation files on-the-fly when they are first needed. To ensure that your benchmark measurements are not skewed by network I/O, it is highly recommended to pre-download all necessary data before starting the benchmark. 

Run the provided script to cache all data locally:
```bash
python download_reps.py
```

### 3. Run the Benchmark

The primary one-thread comparison run with parameters optimized for the 32 GiB workstation:

```bash
python benchmark_tensorpow.py \
  --output results/tensorpow_i5_14600K_1thread \
  --n-values 2:30 \
  --repeats 3 \
  --cold-repeats 1 \
  --direct-repeats 1 \
  --threads 1 \
  --direct-memory-gib 20 \
  --direct-max-n 9
```

### Script Parameters

- `--output`: Prefix path for output files (e.g., `results/run_name`).
- `--n-values`: The tensor powers to evaluate (e.g., `2:30` for range 2 to 30).
- `--repeats`: Number of warm measurements.
- `--cold-repeats`: Number of cold-process measurements.
- `--direct-repeats`: Number of direct Kronecker measurements.
- `--threads`: Number of BLAS/LAPACK threads to use.
- `--direct-memory-gib`: Max memory budget for the direct method before skipping.
- `--direct-max-n`: Max tensor power `n` to evaluate with the direct method.

> [!NOTE]
> For extremely large tensor powers ($n \ge 27$), the script automatically restricts both warm and cold repeats to exactly 1, regardless of the command-line arguments. This prevents excessively long runtimes since timings at these scales are already highly stable and consistent.

### Outputs

Running the script generates continuous updates to the following files:
- `*_raw.csv`: Every timed repetition.
- `*_summary.json`: Medians, absolute deviations, minima, maxima, values, and residuals.
- `*_metadata.json`: Command line and full environment information.
- `*_matrices.npz`: Exact test matrices, seed, and coefficients.

### 4. Process Results (Plots & Tables)

To parse the JSON output into scientific paper quality graphs and tables, a post-processing script is included. It generates formatting-ready `.tex` files (calculating theoretical memory bounds algebraically) alongside high-DPI `.pdf` plots. 

```bash
pip install matplotlib
python process_results.py --input results/benchmark_results_summary.json
```
