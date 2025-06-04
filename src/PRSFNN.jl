module PRSFNN

using Distributions
using Statistics
using LinearAlgebra
using Zygote
using Flux
using Flux: train!
using Random
using StatsBase: sample
using HypothesisTests: ApproximateTwoSampleKSTest
using SnpArrays
using LoggingExtras
using Dates
using SnoopPrecompile  
using Comonicon
using CSV
using DataFrames
using DelimitedFiles
using PDMats
using PDMats: quad
using PDMats: pdadd!
using TimerOutputs
using BSON
using BSON: @load, @save
using LinearSolve
# using Plots
using StatsFuns
using SpecialFunctions
using LogExpFunctions
using Parquet2

# include in logging call, e.g., @info "$(ltime()) message"
function ltime()
    date_format = "yyyy-mm-dd HH:MM:SS"
    return Dates.format(now(), date_format)
end

include("loss.jl")
include("train.jl")
include("simulate.jl")
include("LD.jl")
include("main.jl")
include("load_annot_ss.jl")
include("utils.jl")
include("nn.jl")

export joint_lob_prob, RSS, elbo, train_until_convergence, main, fit_genome_wide_nn, train_cavi, infer_σ2

# speed up precompilation for end users
#@precompile_setup begin
#    @precompile_all_calls begin
#        raw = simulate_raw(;N = 1_000, P = 30, K = 10)
#        ss = estimate_sufficient_statistics(raw[1], raw[3])
#    end
#end

end
