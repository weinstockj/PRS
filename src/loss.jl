


"""
    log_prior(β, σ2_β, p_causal, σ2, spike_σ2, to)

Calculate the log density of β based on a spike and slab prior.

# Arguments
- `β::Vector`: Vector of effect sizes.
- `σ2_β::Vector`: Vector of prior variances for effect sizes.
- `p_causal::Vector`: Vector of prior probabilities that each SNP is causal.
- `σ2::Real`: Global variance parameter.
- `spike_σ2::Real`: Variance parameter for the spike component.
- `to`: Timer object for benchmarking.

# Returns
- `Float64`: Log prior density.
"""
function log_prior(β::Vector, σ2_β::Vector, p_causal::Vector, σ2::Real, spike_σ2::Real, to) #; spike_σ2 = 1e-8) 

    P = length(β)
    slab_dist = Normal.(0, sqrt(σ2) .* sqrt.(σ2_β .+ spike_σ2))
    spike_dist = Normal(0, sqrt(σ2) .* sqrt(spike_σ2))
    # gen = ([logpdf(slab_dist[i], β[i]) + log(p_causal[i]), logpdf(spike_dist, β[i]) + log(1.0 - p_causal[i])] for i in 1:P)
    logprob = 0.0 
    container = zeros(2)
    @timeit to "calculate logsumexp loop" @inbounds @fastmath for i in 1:P
        x = β[i]
        p = p_causal[i]
        container[1] = logpdf(slab_dist[i], x) + log(p)
        container[2] = logpdf(spike_dist, x) + log(1.0 - p)
        logprob += Flux.logsumexp(container)
    end 
    
    # return @fastmath sum(logsumexp.(gen))
    return sum(logprob)
end


#=
function rss(β::Vector, coef::Vector, SE::Vector, R::AbstractArray, to; λ = 1e-8)
    # .000349, 23 allocations no turbo with P = 100

    # .01307 with P = 500
    # S = Matrix(Diagonal(SE))
   # Sinv = Matrix(Diagonal(1 ./ SE)) # need to wrap in Matrix for ReverseDiff
    μ = @timeit to "update μ within rss"  (SE .* R .* (1 ./ SE)') * β
    Σ = @timeit to "create Σ" Hermitian(SE .* R .* SE')
    #println(Σ)
    # there are some very small negative eigenvalues
    # very close to zero in the covariance matrix Σ
    # adding a small positive value to the diagonal of Σ
    # for regularization to make the matrix positive definite
    # Add λ to the diagonal of Σ
    Σ_reg = @timeit to "add ϵ to diagonal Σ" Σ + λ * I
    # Σ_reg = @timeit to "add ϵ to diagonal Σ" Σ 
    # Σ_reg = @timeit to "POET cov " poet_cov(Σ) 
    # dist = MvNormal(μ, Σ)
    val = @timeit to "calculate logpdf" logpdf(MvNormal(μ, Σ_reg), coef)
    #
    return val
    #return logpdf(MvNormal(μ, Σ), coef)
    #return μ, Σ
end
=#

"""
    rss(β, coef, Σ, SRSinv, to)

Calculate the summary statistic RSS likelihood.

# Arguments
- `β::Vector`: Vector of effect sizes.
- `coef::Vector`: Observed coefficients.
- `Σ::AbstractPDMat`: Positive definite covariance matrix.
- `SRSinv::Matrix`: Precomputed matrix for efficiency.
- `to`: Timer object for benchmarking.

# Returns
- `Float64`: Log likelihood value.

# Example
```julia
rss(
    [0.0011, 0.0052, 0.0013],
    [-0.019, 0.013, -0.0199],
    PDMat(Σ),
    SRSinv,
    TimerOutput()
)
```
"""
function rss(β::Vector, coef::Vector, Σ::AbstractPDMat, SRSinv::Matrix, to)

    μ = @timeit to "update μ within rss" @fastmath SRSinv * β
    val = @timeit to "calculate logpdf" @fastmath logpdf(MvNormal(μ, Σ), coef)
    return val
end
"""
    joint_log_prob(β, coef, SE, R, σ2_β, p_causal, σ2, [to])

Compute the joint log probability of the model combining likelihood and prior.

# Arguments
- `β::Vector`: Vector of effect sizes.
- `coef::Vector`: Observed coefficients.
- `SE::Vector` or `Σ::AbstractPDMat`: Standard errors or covariance matrix.
- `R::Matrix` or `SRSinv::Matrix`: Correlation matrix or precomputed matrix.
- `σ2_β::Vector`: Vector of prior variances for effect sizes.
- `p_causal::Vector`: Vector of prior probabilities that each SNP is causal.
- `σ2::Real`: Global variance parameter.
- `spike_σ2::Real`: Optional variance parameter for the spike component.
- `to`: Timer object for benchmarking.

# Returns
- `Float64`: Joint log probability.

# Example
```julia
joint_log_prob(
    [0.0011, 0.0052, 0.0013],
    [-0.019, 0.013, -0.0199],
    [0.0098, 0.0098, 0.0102],
    [1.0 0.03 0.017; 0.031 1.0 -0.03; 0.017 -0.02 1.0],
    [0.01, 0.01, 0.01],
    [0.10, 0.10, 0.10],
    0.01,
    TimerOutput()
)
```
"""
joint_log_prob(β::Vector, coef::Vector, SE::Vector, R::Matrix, σ2_β::Vector, p_causal::Vector, σ2::Real, to) = rss(β, coef, SE, R, to) + log_prior(β, σ2_β, p_causal, σ2, to)

joint_log_prob(β::Vector, coef::Vector, Σ::AbstractPDMat, SRSinv::Matrix, σ2_β::Vector, p_causal::Vector, σ2::Real, spike_σ2::Real, to) = rss(β, coef, Σ, SRSinv, to) + log_prior(β, σ2_β, p_causal, σ2, spike_σ2, to)

"""
    elbo(z, q_μ, log_q_var, coef, SE, R, σ2_β, p_causal, σ2, [spike_σ2], to)

Compute the Evidence Lower Bound (ELBO) for variational inference.

# Arguments
- `z::Vector`: Random vector from standard normal distribution.
- `q_μ::Vector`: Mean vector of the variational distribution.
- `log_q_var::Vector`: Log variance vector of the variational distribution.
- `coef::Vector`: Observed coefficients.
- `SE::Vector` or `Σ::AbstractPDMat`: Standard errors or covariance matrix.
- `R::AbstractArray` or `SRSinv::Matrix`: Correlation matrix or precomputed matrix.
- `σ2_β::Vector`: Vector of prior variances for effect sizes.
- `p_causal::Vector`: Vector of prior probabilities that each SNP is causal.
- `σ2::Real`: Global variance parameter.
- `spike_σ2::Real`: Optional variance parameter for the spike component.
- `to`: Timer object for benchmarking.

# Returns
- `Float64`: Computed ELBO value.

# Example
```julia
elbo(
    rand(Normal(0, 1), 3),
    [0.01, -0.003, 0.0018],
    [-9.234, -9.24, -9.24],
    [0.023, -0.0009, -0.0018],
    [0.0094, 0.00988, 0.0102],
    [1.0 0.03 0.017; 0.031 1.0 -0.03; 0.017 -0.02 1.0],
    [0.01, 0.01, 0.01],
    [0.10, 0.10, 0.10],
    0.01,
    TimerOutput()
)
```
"""
function elbo(z::Vector, q_μ::Vector, log_q_var::Vector, coef::Vector, SE::Vector, R::AbstractArray, σ2_β::Vector, p_causal::Vector, σ2::Real, to)
    q_var = @timeit to "q_var" exp.(log_q_var)
    q = @timeit to "q" MvNormal(q_μ, Diagonal(q_var))
    q_sd = @timeit to "q_sd" sqrt.(q_var)
    ϕ = @timeit to "ϕ" q_μ .+ q_sd .* z
    # γ = compute_γ(q_μ, q_var)   
    # jl =  joint_log_prob(γ .* ϕ, coef, SE, R) 
    jl =  @timeit to "joint_log_prob" joint_log_prob(ϕ, coef, SE, R, σ2_β, p_causal, σ2, to) 
    q = @timeit to "logpd" logpdf(q, ϕ)
    # jac = prod(z)
    return (jl - q)
end

function elbo(z::Vector, q_μ::Vector, log_q_var::Vector, coef::Vector, Σ::AbstractPDMat, SRSinv::Matrix, σ2_β::Vector, p_causal::Vector, σ2::Real, spike_σ2::Real, to)
    q_var = @timeit to "q_var" exp.(log_q_var)
    q = @timeit to "q" MvNormal(q_μ, Diagonal(q_var))
    q_sd = @timeit to "q_sd" sqrt.(q_var)
    ϕ = @timeit to "ϕ" q_μ .+ q_sd .* z
    # γ = compute_γ(q_μ, q_var)   
    # jl =  joint_log_prob(γ .* ϕ, coef, SE, R) 
    jl =  @timeit to "joint_log_prob" joint_log_prob(ϕ, coef, Σ, SRSinv, σ2_β, p_causal, σ2, spike_σ2, to) 
    q = @timeit to "logpd" logpdf(q, ϕ)
    # jac = prod(z)
    return (jl - q)
end

