function standardize(matrix)
    return (matrix .- mean(matrix, dims=1)) ./ std(matrix, dims=1)
end

function plot_loss_vs_epochs(train_losses, test_losses, i, epoch_model_taken)
    plot(1:length(train_losses), train_losses, xlabel="Epochs", ylabel="Loss", label="train", title="Loss vs. Epochs, iteration $i, best at $epoch_model_taken", reuse=false)
    plot!(1:length(test_losses), test_losses, lc=:orange, label="test")
    vline!([epoch_model_taken], label="epoch of best")
end


"""
    log_prior(β, σ2_β, p_causal)
    Calculates the log density of β based on a spiek and slab prior
"""
function log_prior(β::Vector, σ2_β::Vector, p_causal::Vector)

    P = length(β)
    # prob_slab = 0.10
    # L = prob_slab * 1_000
    # h2 = 0.10
    spike_σ2 = 1e-6
    slab_dist = Normal.(0, sqrt.(σ2_β .+ spike_σ2))
    spike_dist = Normal(0, sqrt(spike_σ2))
    logprobs = log.(pdf.(slab_dist, β) .* p_causal .+ pdf.(spike_dist, β) .* (1 .- p_causal))
    return sum(logprobs)
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
function rss(β::Vector, coef::Vector, SE::Vector, R::AbstractArray)
    # .000349, 23 allocations no turbo with P = 100

    # .01307 with P = 500
    # S = Matrix(Diagonal(SE))
   # Sinv = Matrix(Diagonal(1 ./ SE)) # need to wrap in Matrix for ReverseDiff
    μ =  (SE .* R .* (1 ./ SE)') * β
    Σ = Hermitian(SE .* R .* SE')
    #println(Σ)
    # dist = MvNormal(μ, Σ)
    return logpdf(MvNormal(μ, Σ), coef)
    #return μ, Σ
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
joint_log_prob(β, coef, SE, R, σ2_β, p_causal) = rss(β, coef, SE, R) + log_prior(β, σ2_β, p_causal)

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
function elbo(z::Vector, q_μ::Vector, log_q_var::Vector, coef::Vector, SE::Vector, R::AbstractArray, σ2_β::Vector, p_causal::Vector)
    q_var = exp.(log_q_var)
    q = MvNormal(q_μ, Diagonal(I * q_var))
    q_sd = sqrt.(q_var)
    ϕ = q_μ .+ q_sd .* z
    # γ = compute_γ(q_μ, q_var)   
    # jl =  joint_log_prob(γ .* ϕ, coef, SE, R) 
    jl =  joint_log_prob(ϕ, coef, SE, R, σ2_β, p_causal) 
    q = logpdf(q, ϕ)
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


