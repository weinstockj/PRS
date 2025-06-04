## PRSFNN

PRSFNN (Polygenic Risk Score with Functional Neural Networks) is a Julia module for calculating polygenic risk scores by integrating GWAS summary statistics with functional annotations using neural networks.

## Features

- Integration of GWAS summary statistics with functional annotations
- Linkage disequilibrium (LD) calculation and correction
- Coordinate Ascent Variational Inference (CAVI) for posterior effect size estimation
- Neural network models to learn the relationship between functional annotations and genetic effect sizes

## Installation

```julia
# From the Julia REPL
using Pkg
Pkg.add(url="https://github.com/weinstockj/PRS.jl")
```

## Getting Started

Steps to load this module from the root directory:

1. Run `julia --color=yes --project=.` (requires Julia 1.9.0 or later)
2. Run `using Revise` # helpful while developing
3. Run `using PRSFNN`

Now the functions have been loaded. 
To call an internal function, use `PRSFNN.function_name` 

To run Julia in debugger mode: `JULIA_DEBUG=PRSFNN julia --color=yes --project=.`

## Usage

### Basic Example

```julia
using PRSFNN

# Run PRSFNN on a genomic region
result = main(
    output_prefix = "chr3_block1",
    annot_data_path = "path/to/annotations.parquet", 
    gwas_data_path = "path/to/gwas_stats.tsv",
    ld_panel_path = "path/to/ld_panel"
)
```

### Simulating GWAS Data

For testing and development, you can simulate GWAS summary statistic data:

```julia
# Generate simulated data
raw = simulate_raw()
# Extract sufficient statistics needed for PRS
ss = estimate_sufficient_statistics(raw[1], raw[3])

# Visualize the true effect size distribution
using Plots
histogram(raw[2], title="True β Distribution")
```

### Training a PRS Model

```julia
# Train the PRS model
prs_result = train_until_convergence(
    ss[1],         # Standardized beta coefficients
    ss[2],         # Standard errors
    ss[4],         # LD matrix
    ss[5],         # XtX matrix
    raw[6]         # Annotations
)
```

## Documentation

For more detailed documentation, visit the [official documentation site](https://weinstockj.github.io/PRS/dev).

## Running the Tests

Run unit tests with:

```julia
julia --project=. test/runtests.jl
```

or interactively with:

```julia
includet("test/runtests.jl")
```

## Contact

Please address correspondence to:
- Josh Weinstock <josh.weinstock@emory.edu>
- April Kim <aprilkim@jhu.edu>
- Alexis Battle <ajbattle@jhu.edu>

## Functions

```@docs
PRSFNN.main
PRSFNN.rss 
PRSFNN.elbo
PRSFNN.joint_log_prob
PRSFNN.train_until_convergence
PRSFNN.fit_heritability_nn
PRSFNN.log_prior
PRSFNN.estimate_sufficient_statistics
PRSFNN.compute_LD
PRSFNN.fit_genome_wide_nn
PRSFNN.train_cavi
PRSFNN.simulate_raw
PRSFNN.infer_σ2
PRSFNN.poet_cov
```
