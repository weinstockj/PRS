Steps to load this module from the root directory:

1. Run julia (this code expects 1.9.0)
2. Activate the enviroment by typing `]` and then `activate .` Then backspace to return to the Julia REPL
3. Run `using Revise`
4. Run `includet("scratch.jl")`

Now the functions have been sourced. 

## Simulating GWAS data

Then first simulate GWAS summary statistic data:

1. `raw = simulate_raw()`
2. `ss = estimate_sufficient_statistics(raw[1], raw[3])`

Check out the true $\beta$ distribution with:
`using Plots; histogram(raw[2])` 

## Train PRS

`test_new = train_until_convergence(ss[1], ss[2], ss[4], ss[5], raw[5], raw[6], raw[2])`

## Unit testing

Run unit tests with `include("test/runtests.jl")`
