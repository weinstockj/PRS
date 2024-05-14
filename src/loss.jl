function standardize(matrix)
    return (matrix .- mean(matrix, dims=1)) ./ std(matrix, dims=1)
end


"""
    log_prior(β, σ2_β, p_causal)
    Calculates the log density of β based on a spike and slab prior
"""
function log_prior(β::Vector, σ2_β::Vector, p_causal::Vector, to; spike_σ2 = 1e-8)

    P = length(β)
    slab_dist = Normal.(0, sqrt.(σ2_β .+ spike_σ2))
    spike_dist = Normal(0, sqrt(spike_σ2))
    # gen = ([logpdf(slab_dist[i], β[i]) + log(p_causal[i]), logpdf(spike_dist, β[i]) + log(1.0 - p_causal[i])] for i in 1:P)
    logprob = 0.0 
    container = zeros(2)
    @timeit to "calculate logsumexp loop" @inbounds @fastmath for i in 1:P
        x = β[i]
        p = p_causal[i]
        container[1] = logpdf(slab_dist[i], x) + log(p)
        container[2] = logpdf(spike_dist, x) + log(1.0 - p)
        logprob += logsumexp(container)
    end 
    
    # return @fastmath sum(logsumexp.(gen))
    return sum(logprob)
end

"""
    rss(β, coef, SE, R)
    Calculate the summary statistic RSS likelihood


```julia-repl
rss(
    [0.0011, .0052, 0.0013],
    [-0.019, 0.013, -.0199],
    [.0098, .0098, .0102],
    [1.0 .03 .017; .031 1.0 -0.03; .017 -0.02 1.0]
)
```
"""
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

function rss(β::Vector, coef::Vector, Σ::AbstractPDMat, SRSinv::Matrix, to)

    μ = @timeit to "update μ within rss" @fastmath SRSinv * β
    val = @timeit to "calculate logpdf" @fastmath logpdf(MvNormal(μ, Σ), coef)
    return val
end
"""
`joint_log_prob(β, coef, SE, R, σ2_β, p_causal)`

Compute the joint log probability of the model

```julia-repl

joint_log_prob(
    [0.0011, .0052, 0.0013],
    [-0.019, 0.013, -.0199],
    [.0098, .0098, .0102],
    [1.0 .03 .017; .031 1.0 -0.03; .017 -0.02 1.0],
    0.01,
    0.10
)
```
"""
joint_log_prob(β::Vector, coef::Vector, SE::Vector, R::Matrix, σ2_β::Vector, p_causal::Vector, to) = rss(β, coef, SE, R, to) + log_prior(β, σ2_β, p_causal, to)

joint_log_prob(β::Vector, coef::Vector, Σ::AbstractPDMat, SRSinv::Matrix, σ2_β::Vector, p_causal::Vector, to) = rss(β, coef, Σ, SRSinv, to) + log_prior(β, σ2_β, p_causal, to)

"""
    elbo(z, q_μ, log_q_var, coef, SE, R, σ2_β, p_causal)


```julia-repl
elbo(
    rand(Normal(0, 1), 3),
    [0.01, -0.003, 0.0018],
    [-9.234, -9.24, -9.24],
    [0.023, -0.0009, -.0018],
    [.0094, .00988, .0102],
    [1.0 .03 .017; .031 1.0 -0.03; .017 -0.02 1.0],
    0.01,
    0.10
)
```
"""
function elbo(z::Vector, q_μ::Vector, log_q_var::Vector, coef::Vector, SE::Vector, R::AbstractArray, σ2_β::Vector, p_causal::Vector, to)
    q_var = @timeit to "q_var" exp.(log_q_var)
    q = @timeit to "q" MvNormal(q_μ, Diagonal(q_var))
    q_sd = @timeit to "q_sd" sqrt.(q_var)
    ϕ = @timeit to "ϕ" q_μ .+ q_sd .* z
    # γ = compute_γ(q_μ, q_var)   
    # jl =  joint_log_prob(γ .* ϕ, coef, SE, R) 
    jl =  @timeit to "joint_log_prob" joint_log_prob(ϕ, coef, SE, R, σ2_β, p_causal, to) 
    q = @timeit to "logpd" logpdf(q, ϕ)
    # jac = prod(z)
    return (jl - q)
end

function elbo(z::Vector, q_μ::Vector, log_q_var::Vector, coef::Vector, Σ::AbstractPDMat, SRSinv::Matrix, σ2_β::Vector, p_causal::Vector, to)
    q_var = @timeit to "q_var" exp.(log_q_var)
    q = @timeit to "q" MvNormal(q_μ, Diagonal(q_var))
    q_sd = @timeit to "q_sd" sqrt.(q_var)
    ϕ = @timeit to "ϕ" q_μ .+ q_sd .* z
    # γ = compute_γ(q_μ, q_var)   
    # jl =  joint_log_prob(γ .* ϕ, coef, SE, R) 
    jl =  @timeit to "joint_log_prob" joint_log_prob(ϕ, coef, Σ, SRSinv, σ2_β, p_causal, to) 
    q = @timeit to "logpd" logpdf(q, ϕ)
    # jac = prod(z)
    return (jl - q)
end

function compute_γ(q_μ, q_var; slab_σ = sqrt(0.10 / 10 + 1e-6), p_causal = 0.10)

    q_sd = sqrt.(q_var)
    
    SSR = (q_μ .^ 2) ./ q_var
    odds = (p_causal / (1 - p_causal)) .* (q_sd ./ slab_σ) .* exp.(SSR ./ 2)
    γ = odds ./ (odds .+ 1)
    
    return γ
end

σ(x) = 1.0 ./ (1.0 .+ exp.(-1.0 .* x))

