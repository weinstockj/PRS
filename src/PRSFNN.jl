module PRSFNN

using LoopVectorization
using Distributions
using Statistics
using LinearAlgebra
using Zygote
using Flux
using Flux: train!
using Plots
using Random
using StatsBase: sample
using HypothesisTests: ApproximateTwoSampleKSTest
using SnpArrays
using LoggingExtras
using Dates
using SnoopPrecompile  

# include in logging call, e.g., @info "$(ltime()) message"
function ltime()
    date_format = "yyyy-mm-dd HH:MM:SS"
    return Dates.format(now(), date_format)
end

include("loss.jl")
include("train.jl")
include("simulate.jl")

export joint_lob_prob, RSS, elbo, train_until_convergence

# speed up precompilation for end users
@precompile_setup begin
    @precompile_all_calls begin
        raw = simulate_raw()
        ss = estimate_sufficient_statistics(raw[1], raw[3])
    end
end

end
