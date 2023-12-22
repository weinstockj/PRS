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

include("loss.jl")
include("train.jl")
include("simulate.jl")

export joint_lob_prob, RSS, elbo, train_until_convergence

end
