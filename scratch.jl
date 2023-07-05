using MKL
using LoopVectorization
using Distributions
using Statistics
using LinearAlgebra
using Zygote
using Flux
using Flux: train!
# import AbstractDifferentiation as AD
# using Enzyme
# using ReverseDiff: GradientTape, gradient, gradient!, compile, GradientConfig, DiffResults

# P is number of SNPs, K is number of annotations
# output is normalized variance per SNP e.g., s * h2 / P where h2 / P 
# is the average SNP and s is a small adjustment
function draw_slab_s_given_annotations(P; K = 100)
   
    G = rand(Normal(0, 1.0), P, K) # matrix of annotations - could be made more realistic?
   
    # three possibly ways in which the annotations informs per SNP h2
    functions = (
        x -> 0.10 .* x, # linear
        x -> 0.02 .* x .^ 2, # quadratic,
        x -> x .* 0, # the annotation does nothing!
    )

    choose_f = Categorical([.3, .05, .65])
    # pick 1 of the 3 functions
    choices = [rand(choose_f) for i in 1:K]
    ϕ = [functions[choices[i]] for i in 1:K] # randomly pick a linear or quadratic transformation
    
    σ2 = zeros(P)
    for i in 1:P
        # define variance of SNPs as sum of all of the ϕ(G) + one interaction term between 
        # the first two annotations to throw in some complexity
        σ2[i] = exp(sum([ϕ[j](G[i, j]) for j in 1:K]) + ϕ[1](G[i, 1]) * ϕ[2](G[i, 2]))
    end

    s = σ2 ./ mean(σ2) # normalize by average variance

    return s, G, choices
end

# how many epochs do we need?
# is momentum the right choice here as optimizer?
function fit_heritability_nn(s, G; n_epochs = 400)

    K = size(G, 2)
    P = size(G, 1)
    H = 3
    # super simple neural network with one hidden layer with 5 nodes
    model = Chain(
        Dense(K => H, relu; init = Flux.glorot_normal(gain = 0.0005)),
        Dense(H => 1)
    )

    # RMSE
    function loss(model, x, y)
        yhat = vec(model(transpose(x))) # need vec to convert 1 x P matrix to length P vec
        Flux.mse(yhat, y)
    end

    opt = Flux.setup(Momentum(), model)
    data = [
            (
            Float32.(G), # to address Float64 -> Float32 Flux complaint
            log1p.(s), # later apply inverse of exp.(x) .- 1.0
        )
        ] 

    for epoch in 1:n_epochs
        train!(loss, model, data, opt)

        println("epoch = $epoch")
        println(loss(model, G, s))
    end
    
    return model
end

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

    # Simulation annotations and create expected s parameter for causal SNPs
    s, G, _ = draw_slab_s_given_annotations(sum(γ))

    spike = rand(Normal(0, 0.001), P)
    slab  = rand(Normal(0,  sqrt(h2 / L)), P)
    β = γ .* slab + (1 .- γ) .* spike
    β[γ] .= β[γ] .* sqrt.(s)
    println(β)
    μ = X * β
    vXβ = var(μ)
    # h2 = (var(XB)) / (var(XB) + noise)
    # h2 * var(XB) + h2 * noise = var(XB)
    # h2 * noise = var(XB) * (1 - h2)
    # noise = var(XB) * (1 - h2) / h2
    σ2 = vXβ * (1 - h2) / h2

    Y = μ + rand(Normal(0, sqrt(σ2)), N)
    return X, β, Y, Σ, s, G
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
    return coef, SE, Z, cor(X), D
end

function log_prior(β)

    P = length(β)
    prob_slab = 0.10
    L = prob_slab * 1_000
    h2 = 0.10
    spike_σ2 = 1e-6
    slab_dist = Normal(0, sqrt(h2 / L + spike_σ2))
    spike_dist = Normal(0, sqrt(spike_σ2))
    logprobs = log.(pdf.(slab_dist, β) .* prob_slab .+ pdf.(spike_dist, β) .* (1 - prob_slab))
    return sum(logprobs)
end

"""
    elbo(z, q_μ, log_q_var, coef, SE, R)

```julia-repl
rss(
    [0.0011, .0052, 0.0013],
    [-0.019, 0.013, -.0199],
    [.0098, .0098, .0102],
    [1.0 .03 .017; .031 1.0 -0.03; .017 -0.02 1.0]
)
```
"""
function rss(β, coef, SE, R)
    # .000349, 23 allocations no turbo with P = 100

    # .01307 with P = 500
    # S = Matrix(Diagonal(SE))
   # Sinv = @turbo Matrix(Diagonal(1 ./ SE)) # need to wrap in Matrix for ReverseDiff
    μ =  @turbo (SE .* R .* (1 ./ SE)') * β
    Σ = @turbo Hermitian(SE .* R .* SE')
    #println(Σ)
    # dist = MvNormal(μ, Σ)
    return logpdf(MvNormal(μ, Σ), coef)
    #return μ, Σ
end

"""
```julia-repl
joint_log_prob(
    [0.0011, .0052, 0.0013],
    [-0.019, 0.013, -.0199],
    [.0098, .0098, .0102],
    [1.0 .03 .017; .031 1.0 -0.03; .017 -0.02 1.0]
)
```
"""
joint_log_prob(β, coef, SE, R) = rss(β, coef, SE, R) + log_prior(β)

"""
    elbo(z, q_μ, log_q_var, coef, SE, R)

```julia-repl
elbo(
    rand(Normal(0, 1), 3),
    [0.01, -0.003, 0.0018],
    [-9.234, -9.24, -9.24],
    [0.023, -0.0009, -.0018],
    [.0094, .00988, .0102],
    [1.0 .03 .017; .031 1.0 -0.03; .017 -0.02 1.0]
)
```
"""
function elbo(z, q_μ, log_q_var, coef, SE, R)
    q_var = exp.(log_q_var)
    q = MvNormal(q_μ, Diagonal(I * q_var))
    q_sd = sqrt.(q_var)
    ϕ = q_μ .+ q_sd .* z
    # γ = compute_γ(q_μ, q_var)   
    # jl =  joint_log_prob(γ .* ϕ, coef, SE, R) 
    jl =  joint_log_prob(ϕ, coef, SE, R) 
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

function train_block(q_μ, q_var, coef::AbstractVector, SE::AbstractVector, R::AbstractMatrix; max_iter = 20, N = 10_000, n_elbo = 20)

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
            # P_samples = rand.(Bernoulli.(q_p))
            l, grad = Zygote.withgradient(
                (a, b, c, d, e) -> elbo(
                    z, # random Z\
                    # P_samples,
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


"""
```julia-repl
train_cavi(
    ss[1],
    ss[2],
    ss[4],
    ss[5]
)
```
"""
function train_cavi(coef, SE, R, D; n_elbo = 50, max_iter = 20, N = 10_000, σ2_β = .01, σ2 = 1.0, p_causal = 0.01)
   
    # TODO: eventually, replace fixed slab variance and inclusion probabilty with 
    # annotation informed prior

    P = length(coef)

    q_μ = zeros(P)
    q_var = ones(P) * 0.001
    q_sd = sqrt.(q_var)
    q_α = ones(P) .* 0.10
    q_odds = ones(P) 
    SSR = ones(P)

    X_sd = sqrt.(D ./ N)
    Xty = coef .* D
    XtX = @turbo Diagonal(X_sd) * R * Diagonal(X_sd) .* N

    @inbounds for i in 1:max_iter

        # if clause just to monitor loss convergence
        if (mod(i, 5) == 0) | (i == 1)
            loss = 0.0
            @inbounds for z in 1:n_elbo
                loss = loss + elbo(rand(Normal(0, 1), P), q_μ, log.(q_var), coef, SE, R)
            end
         
            loss = loss / n_elbo
            println("i = $i, loss = $loss (bigger numbers are better)")   
        end

        @turbo q_var .= σ2 ./ (diag(XtX) .+ 1 / σ2_β)
        @inbounds for k in 1:P
            J = setdiff(1:P, k)
            q_μ[k] = (view(q_var, k) ./ σ2) .* (view(Xty, k) .- sum(view(XtX, k, J) .* view(q_α, J) .* view(q_μ, J)))
        end
        @turbo SSR .= q_μ .^ 2 ./ q_var
        @turbo q_odds .= (p_causal / (1 - p_causal)) .* q_sd ./ sqrt(σ2_β) .* exp.(SSR ./ 2.0)
        @turbo q_α .= q_odds ./ (1.0 .+ q_odds)


    end

    return q_μ, q_α, q_var
end

function train(coef::Vector{Float64}, SE::Vector{Float64}, R::AbstractMatrix)
    P = length(coef)

    # β = rand(Normal(0, .001), P)
    q_μ = rand(Normal(0, .1), P) # init q
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
            n_elbo = 20
        )
        # loss_vector[i] = -1.0 joint_log_prob(q_μ, coef, SE, R)
        loss_vector[i] = last(loss)
    end

    γ = compute_γ(q_μ, exp.(q_var))

    return q_μ, γ, exp.(q_var), loss_vector
end