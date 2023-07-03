Steps to load this module from the root directory:

1. Run julia (this code expects 1.9.0)
2. Activate the enviroment by typing `]` and then `activate .` Then backspace to return to the Julia REPL
3. Run `using Revise`
4. Run `includet("scratch.jl")`

Now the functions have been sourced. 

Then first simulate GWAS summary statistic data:

1. `raw = simulate_raw()`
2. `ss = estimate_sufficient_statistics(raw[1], raw[3])`

Check out the true $\beta$ distribution with:
`using Plots; histogram(raw[2])` 

Now train the PRS with ADVI:
 `@time out = train(ss[1], ss[2], ss[4])`

 Or use CAVI updates from Carbonetto and Stephens 2012
 `@time cavi_out = train_cavi(ss[1], ss[2], ss[4], ss[5])` 

The CAVI posterior means are indeed quite sparse:

`histogram(cavi_out[1] .* cavi_out[2])`

To compare the posterior means with the true coefficients:

1. `scatter(out[1], raw[2])`
2. `cor(out[1], raw[2]) ^ 2`

We can compare the CAVI and ADVI output with

`scatter(cavi_out[1], out[1])` and
`scatter(cavi_out[1] .* cavi_out[2], out[1] .* out[2])`