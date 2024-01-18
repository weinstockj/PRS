## PRSFNN

Steps to load this module from the root directory:

1. Run `julia --color=yes --project=.` (this code expects 1.9.0)
3. Run `using Revise`
4. Run `using PRSFNN` # takes 32 seconds on my machine

Now the functions have been sourced. 

## Simulating GWAS data

Then first simulate GWAS summary statistic data:

1. `raw = simulate_raw()`
2. `ss = estimate_sufficient_statistics(raw[1], raw[3])`

Check out the true $\beta$ distribution with:
`using Plots; histogram(raw[2])` 

## Train PRS

`test_new = train_until_convergence(ss[1], ss[2], ss[4], ss[5], raw[6])`

## Unit testing

Run unit tests with `includet("test/runtests.jl")`

## Functions

```@docs
PRSFNN.main
PRSFNN.rss 
PRSFNN.elbo
PRSFNN.joint_log_prob
PRSFNN.train_until_convergence
PRSFNN.log_prior
PRSFNN.estimate_sufficient_statistics
PRSFNN.compute_LD
PRSFNN.comonicon_install
PRSFNN.comonicon_install_path
```

