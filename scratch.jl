using MKL
using LoopVectorization
using Distributions
using Statistics
using LinearAlgebra
using Zygote
# import AbstractDifferentiation as AD
# using Enzyme
# using ReverseDiff: GradientTape, gradient, gradient!, compile, GradientConfig, DiffResults

function simulate_raw()
    N = 10_000
    P = 1_000
    eigenvalues = sort(rand(Exponential(1), P))
    eigenvalues[length(eigenvalues)] = eigenvalues[length(eigenvalues)] * 10
    D = Diagonal(eigenvalues)
    println(eigenvalues)
    U = rand(Normal(0, 1), P, P)
    U, _ = qr(U)
    Σ = U * D * U'
    Σ = 0.5 * (Σ + Σ')
    X = rand(MvNormal(zeros(P), Σ), N)'
    p_causal = 0.10
    L = p_causal * P # 100
    γ = rand(Bernoulli(p_causal), P)
    h2 = 0.10
    s = rand(Normal(1, 0.1), P)
    spike = rand(Normal(0, 0.001), P)
    slab  = rand(Normal(0,  sqrt(h2 / L)), P) .* sqrt.(s)
    β = γ .* slab + (1 .- γ) .* spike
    println(β)
    μ = X * β
    vXβ = var(μ)
    # h2 = (var(XB)) / (var(XB) + noise)
    # h2 * var(XB) + h2 * noise = var(XB)
    # h2 * noise = var(XB) * (1 - h2)
    # noise = var(XB) * (1 - h2) / h2
    σ2 = vXβ * (1 - h2) / h2

    Y = μ + rand(Normal(0, sqrt(σ2)), N)
    return X, β, Y, Σ
end

function estimate_sufficient_statistics(X, Y)
    N = size(X, 1)
    P = size(X, 2)
    D = map(x -> sum(x .^ 2), eachcol(X))
    coef = X'Y ./ D
    mse = map(i -> var(Y - view(coef, i) .* view(X, :, i)), 1:P)
    SE2 = mse ./ D
    SE = sqrt.(SE2)
    Z = coef ./ SE
    R = cor(X)
    return coef, SE, Z, cor(X)
end

function log_prior(β)

    P = length(β)
    prob_slab = 0.01
    L = prob_slab * 1_000
    h2 = 0.10
    spike_σ2 = 1e-6
    slab_dist = Normal(0, sqrt(h2 / L + spike_σ2))
    spike_dist = Normal(0, sqrt(spike_σ2))
    logprobs = log.(pdf.(slab_dist, β) .* prob_slab .+ pdf.(spike_dist, β) .* (1 - prob_slab))
    return sum(logprobs)
end

function rss(β, coef, SE, R)
    # .000349, 23 allocations no turbo with P = 100

    # .01307 with P = 500
    S = Matrix(Diagonal(SE))
    Sinv = Matrix(Diagonal(1 ./ SE)) # need to wrap in Matrix for ReverseDiff
    μ = S * R * Sinv * β
    Σ = Hermitian(S * R * S)
    #println(Σ)
    dist = MvNormal(μ, Σ)
    return logpdf(dist, coef)
    #return μ, Σ
end


joint_log_prob(β, coef, SE, R) = rss(β, coef, SE, R) + log_prior(β)

"""
    elbo(z, q, coef, SE, R)

TBW
"""
function elbo(z, q_μ, log_q_var, coef, SE, R)
    q_var = exp.(log_q_var)
    q = MvNormal(q_μ, Diagonal(I * q_var))
    q_sd = sqrt.(q_var)
    ϕ = q_μ .+ q_sd .* z
    jl =  joint_log_prob(ϕ, coef, SE, R) 
    q = logpdf(q, ϕ)
    # jac = prod(z)
    return (jl - q)
end

σ(x) = 1.0 ./ (1.0 .+ exp.(-1.0 .* x))

function train_block(q_μ, q_var, q_p, coef::AbstractVector, SE::AbstractVector, R::AbstractMatrix; max_iter = 20, N = 10_000, n_elbo = 20)

    P = length(coef)
    # 0.08 seconds with zygote, 107k allocations, 20 Mib
    # 0.88 seconds (3.36M allocations, 162 MiB)
    # with P = 500, 0.69 seconds with Zygote and MKL
    # f_tape = GradientTape((a, b, c, d) -> joint_log_prob(a, b, c, d) / N, (rand(P), rand(P), rand(Uniform(0.01, 0.1), P), Matrix(I * 1.0, P, P)))
    # compiled_f_tape = compile(f_tape)
    # inputs = (β, coef, SE, R)
    # results = (similar(β), similar(β), similar(β), similar(R))
    # all_results = map(DiffResults.GradientResult, results)

    loss = Vector{Float64}()
    # η = 0.018
    η = 0.3
    η_μ = ones(P) * .05
    η_var = ones(P) * .05
    s_μ = ones(P) * .001
    s_μ_prev = ones(P) * .001
    s_var = ones(P) * .001
    s_var_prev = ones(P) * .001
    τ = 1.0
    α = 0.3
    last_i = 1
    std_normal = Normal(0, 1)


    @inbounds for i in 1:max_iter
        last_i = i
        if i % 10 == 0
            η = η * 0.98 #reduce learning rate as time moves on
        end

        # grad = gradient!(results, compiled_f_tape, inputs)
        # grad = Zygote.gradient((a, b, c, d) -> joint_log_prob(a, b, c, d) / N, β, coef, SE, R)
        grad_q_μ = zeros(P)
        grad_q_var = zeros(P)
        grad_q_p = zeros(P)
        l = 0.0
        z = zeros(P)
        @inbounds for j in 1:n_elbo # number of elbo samples
            z = rand(std_normal, P)
            P_samples = rand.(Bernoulli.(q_p))
            l, grad = Zygote.withgradient(
                (a, b, c, d, e) -> elbo(
                    z, # random Z\
                    P_samples,
                    a,
                    b,
                    c, 
                    d, 
                    e) / N, 
                q_μ, 
                q_var, 
                coef, 
                SE, 
                R
            )
            grad_q_μ .= grad_q_μ .+ grad[1]
            # grad_q_var .= grad_q_var .+ grad[2] .* exp.(q_var)
            grad_q_var .= grad_q_var .+ grad[2] 

        end
        push!(loss, l)
        #println("now in iter $i")
    #    inputs[1] .= inputs[1] .+ η .* results[1]
        # println("grad_q_var = $grad_q_var")
        # println("s_var = $s_var")

        # β .= β .+ η .* grad[1]
        s_μ .= α .* grad_q_μ .^ 2 + (1.0 - α) .* s_μ_prev
        s_var .= α .* grad_q_var .^ 2 + (1.0 - α) .* s_var_prev
        η_μ .= η ./ (τ .+ sqrt.(s_μ))
        η_var .= 10 .* η ./ (τ .+ sqrt.(s_var))
        q_μ .= q_μ .+ η_μ .* grad_q_μ / n_elbo
        q_var .= q_var .+ η_var .* grad_q_var / n_elbo

        if (norm(grad_q_μ, 2) / P) < 0.0001
            break
        end

        #loss[i] = -1.0 * joint_log_prob(inputs...)
        #println("now β")
    end
    println("ending at iter $last_i")
    # println("$loss")

    return loss
end

function train(coef::Vector{Float64}, SE::Vector{Float64}, R::AbstractMatrix)
    P = length(coef)

    # β = rand(Normal(0, .001), P)
    q_μ = rand(Normal(0, .0001), P) # init q
    q_var = log.(ones(P) * 0.0001) # on log scale

    N_BLOCKS = 20
    BLOCK_LEN = P ÷ N_BLOCKS
    loss_vector = zeros(N_BLOCKS)

    for i in 1:N_BLOCKS
        println("Now on block $i of $N_BLOCKS")
        s = (i - 1) * BLOCK_LEN + 1
        e = i * BLOCK_LEN
        loss = train_block(
            # view(β, s:e), 
            view(q_μ, s:e),
            view(q_var, s:e),
            view(coef, s:e), 
            view(SE, s:e), 
            view(R, s:e, s:e); 
            max_iter = 100,
            n_elbo = 10
        )
        # loss_vector[i] = -1.0 joint_log_prob(q_μ, coef, SE, R)
        loss_vector[i] = last(loss)
    end

    return q_μ, q_var, loss_vector
end