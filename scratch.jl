using MKL
using LoopVectorization
using Distributions
using Statistics
using LinearAlgebra
using Zygote
using Flux
using Flux: train!
using Plots
using Random
# import AbstractDifferentiation as AD
# using Enzyme
# using ReverseDiff: GradientTape, gradient, gradient!, compile, GradientConfig, DiffResults

# P is number of SNPs, K is number of annotations
# output is normalized variance per SNP e.g., s * h2 / P where h2 / P 
# is the average SNP and s is a small adjustment
function draw_slab_s_given_annotations(P; K = 100)
   
    K_half = K ÷ 2
   
    # G = rand(Normal(0, 1.0), P, K) # matrix of annotations - could be made more realistic?
    G_cont = rand(Normal(0, 1.0), P, K_half)  # half continuous annotations (normally distributed)
    G_binom = rand(Binomial(1, 0.5), P, K_half)  # half binomial annotations
    G = hcat(G_cont, G_binom)  # combine annotations
   
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

function standardize(matrix)
    return (matrix .- mean(matrix, dims=1)) ./ std(matrix, dims=1)
end

function plot_loss_vs_epochs(losses, i)
    plot(1:length(losses), losses, xlabel="Epochs", ylabel="Loss", legend=false, title="Loss vs. Epochs, iteration $i", reuse=false)
end

# how many epochs do we need?
# is momentum the right choice here as optimizer?
## ak: modified to have two outputs, and defined two losses: loss_slab, loss_causal
function fit_heritability_nn(s, p, G, iter; n_epochs = 300, patience=30, mse_improvement_threshold=0.05)

    # function split_activation(x)
    #     # Apply softplus to the first output
    #     y1 = softplus(x[1, :])
    #     # Keep the second output unchanged
    #     y2 = x[2, :]
    #     return vcat(y1', y2')
    # end

    K = size(G, 2)
    P = size(G, 1)
    H = 10
    # super simple neural network with one hidden layer with 5 nodes
    model = Chain(
        Dense(K => H, relu; init = Flux.glorot_normal(gain = 0.0005)),
        # Dense(K => H; init = Flux.glorot_normal(gain = 0.0005)),
        # BatchNorm(H),
        # relu,
        Dense(H => 2)
    )

    weight_slab = 0.1 #0.001
    weight_causal = 1

    # RMSE
    function loss(model, x, y_slab, y_causal) ## ak: need two losses for slab variance and percent causal 
        # yhat = vec(model(transpose(x))) # need vec to convert 1 x P matrix to length P vec
        # Flux.mse(yhat, y)
        yhat = model(transpose(x))
        loss_slab = Flux.mse(yhat[1, :], y_slab)
        weighted_loss_slab = weight_slab * loss_slab
        loss_causal = Flux.mse(yhat[2, :], y_causal)
        weighted_loss_causal = weight_causal * loss_causal
        println("slab var loss = $loss_slab")
        println("slab var loss weighted = $weighted_loss_slab")
        println("percent causal loss = $loss_causal")
        println("percent causal loss weighted = $weighted_loss_causal")
        # return loss_slab + loss_causal ## ak: losses summed to form the total loss for training
        return weighted_loss_slab + weighted_loss_causal ## ak: losses summed to form the total loss for training

    end

    # logit(x) = log((x .+ 1e-10) ./ (1.0 .- x .+ 1e-10)) # adding small constant to prevent from being 0
    logit(x) = log((x) ./ (1.0 .- x)) # adding small constant to prevent from being 0
    G_standardized = standardize(G)

    # println("uh oh")
    # println(log.(s))

    # Momentum()
    # RAdam()
    # AdaDelta()
    opt = Flux.setup(AdaGrad(0.01), model) ## ak: using adaptive gradient; yay for adapting the learning rate on its own!
    data = [
            (
            Float32.(G_standardized), # to address Float64 -> Float32 Flux complaint
            log.(s), # later apply inverse of exp.(x) .- 1.0
            # log.(s), # ak: log won't work for negative expected betas
            logit.(p)
        )
        ]
    println("min, max s")
    println(minimum(s))
    println(maximum(s))
    println("min, max p")
    println(minimum(p))
    println(maximum(p))
    
    best_loss = Inf
    count_since_best = 0
    count_below_threshold = 0
    epoch_losses = []

    for epoch in 1:n_epochs
        println("training!")
        ## Train, update model's weights based on loss f, data, and optimizer
        ## Use backpropogation to compute the gradients of the model's parameters wrt the loss, and then the
        ## optimizer updates the params to minimize this loss
        ## Data passed to train! is used to train the model
        train!(loss, model, data, opt)
        println("epoch = $epoch")
        ## Compute current loss of the model on the entire dataset represented by G, s, p, without updating the model's weights
        ## We want to use this to monitor the models performance on the training set after each epoch
        # current_loss = loss(model, G_standardized, log1p.(s), logit.(p))
        current_loss = loss(model, G_standardized, log.(s), logit.(p))
        push!(epoch_losses, current_loss)
        println("current loss weighted = $current_loss")

        # Check for improvement in loss
        mse_improvement = best_loss - current_loss
        if mse_improvement > mse_improvement_threshold
            best_loss = current_loss
            count_since_best = 0
            count_below_threshold = 0
        else
            count_since_best += 1
            count_below_threshold += 1
        end

        # If no improvement for "patience" epochs or consistent small improvement, stop training
        if count_since_best >= patience || count_below_threshold >= 20
            println("Early stopping after $epoch epochs.")
            break
        end

    end

    # Plotting the loss
    plot_nn_loss = plot_loss_vs_epochs(epoch_losses, iter)
    savefig("~/Downloads/scprs_figs/epoch_losses_$iter.png")
    display(plot_nn_loss)
    
    return model
end

function simulate_raw()

    Random.seed!(0)

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
    # s, G, _ = draw_slab_s_given_annotations(sum(γ))
    s, G, _ = draw_slab_s_given_annotations(P)

    spike = rand(Normal(0, 0.001), P)
    slab  = rand(Normal(0,  sqrt(h2 / L)), P)
    β = γ .* slab + (1 .- γ) .* spike
    # β[γ] .= β[γ] .* sqrt.(s)
    β .= β .* sqrt.(s) ## ak: s is normalized variance per SNP for ALL SNPs, not just causal #*#
    # println(β)
    μ = X * β
    vXβ = var(μ)
    # h2 = (var(XB)) / (var(XB) + noise)
    # h2 * var(XB) + h2 * noise = var(XB)
    # h2 * noise = var(XB) * (1 - h2)
    # noise = var(XB) * (1 - h2) / h2
    σ2 = vXβ * (1 - h2) / h2

    Y = μ + rand(Normal(0, sqrt(σ2)), N)
    return X, β, Y, Σ, s, G, γ
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

function log_prior(β, σ2_β, p_causal)

    P = length(β)
    # prob_slab = 0.10
    # L = prob_slab * 1_000
    # h2 = 0.10
    spike_σ2 = 1e-6
    # slab_dist = Normal(0, sqrt(h2 / L + spike_σ2))
    slab_dist = Normal.(0, sqrt.(σ2_β .+ spike_σ2))
    spike_dist = Normal(0, sqrt(spike_σ2))
    # logprobs = log.(pdf.(slab_dist, β) .* prob_slab .+ pdf.(spike_dist, β) .* (1 - prob_slab))
    logprobs = log.(pdf.(slab_dist, β) .* p_causal .+ pdf.(spike_dist, β) .* (1 .- p_causal))
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
   # Sinv = Matrix(Diagonal(1 ./ SE)) # need to wrap in Matrix for ReverseDiff
    μ =  (SE .* R .* (1 ./ SE)') * β
    Σ = Hermitian(SE .* R .* SE')
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
joint_log_prob(β, coef, SE, R, σ2_β, p_causal) = rss(β, coef, SE, R) + log_prior(β, σ2_β, p_causal)

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
function elbo(z, q_μ, log_q_var, coef, SE, R, σ2_β, p_causal)
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

function train_cavi(q_μ, q_α, q_var, p_causal, σ2_β, X_sd, i_iter, coef, SE, R, D; n_elbo = 50, max_iter = 4, N = 10_000, σ2 = 1.0)
   
    # TODO: eventually, replace fixed slab variance and inclusion probabilty with 
    # annotation informed prior

    P = length(coef)

    # q_μ = zeros(P)
    # q_var = ones(P) * 0.001
    q_sd = sqrt.(q_var)
    # q_α = ones(P) .* 0.10
    q_odds = ones(P) 
    SSR = ones(P)

    # X_sd = sqrt.(D ./ N)
    Xty = coef .* D
    XtX = Diagonal(X_sd) * R * Diagonal(X_sd) .* N

    loss = 0.0
    cavi_loss = []

    @inbounds for i in 1:max_iter

        # if clause just to monitor loss convergence
        if (mod(i, 1) == 0) | (i == 1)
            loss = 0.0
            @inbounds for z in 1:n_elbo
                loss = loss + elbo(rand(Normal(0, 1), P), q_μ, log.(q_var), coef, SE, R, σ2_β, p_causal)
            end
         
            loss = loss / n_elbo

            if isnan(loss) == true
                println("q_μ")
                println(q_μ)
                println("q_var")
                println(q_var)
                println("q_odds")
                println(q_odds)
                println("q_α")
                println(q_α)
                break
            end

            push!(cavi_loss, loss)

            println("i = $i, loss = $loss (bigger numbers are better)")   
        end

        # σ2_β and p_causal are vectors

        q_var .= σ2 ./ (diag(XtX) .+ 1 ./ σ2_β) ## ak: eq 8; \s^2_k; does not depend on alpha and mu from previous
        @inbounds for k in 1:P
            J = setdiff(1:P, k)
            q_μ[k] = (view(q_var, k) ./ σ2) .* (view(Xty, k) .- sum(view(XtX, k, J) .* view(q_α, J) .* view(q_μ, J))) ## ak: eq 9; update u_k
        end
        SSR .= q_μ .^ 2 ./ q_var
        q_odds .= (p_causal ./ (1 .- p_causal)) .* q_sd ./ sqrt.(σ2_β) .* exp.(SSR ./ 2.0) ## ak: eq 10; update a_k 
        q_α .= q_odds ./ (1.0 .+ q_odds)


    end

    # with probability q_alpha, additive effect beta is normal with mean q_mu and variance q_var
    return q_μ, q_α, q_var, q_odds, loss, cavi_loss
end

function plot_max_effect_size_vs_iteration(effect_sizes)
    plot(1:length(effect_sizes), effect_sizes, xlabel="Iteration", ylabel="Maximum absolute effect size", legend=false, title="Max abs effect size at each iteration", reuse=false)
end

function plot_cavi_losses(cavi_loss)
    plot(1:length(cavi_loss), cavi_loss, xlabel="Iteration", ylabel="Maximum absolute effect size", legend=false, title="Max abs effect size at each iteration", reuse=false)
end

function plot_corr_true_estimated(true_estimated_corr)
    plot(1:length(true_estimated_corr), true_estimated_corr, xlabel="Iteration", ylabel="Correlation (r)", legend=false, title="Correlation of true and est betas at each iteration", reuse=false)
end

# coef, SE, Z, cor(X), D
function train_until_convergence(coef, SE, R, D, s, G, true_betas; max_iter = 3, threshold = 0.1, N = 10_000) # max_iter = 30, 

    ## initialize
    P = length(coef)
    q_μ = zeros(P)
    q_α = ones(P) .* 0.10
    q_var = ones(P) * 0.001

    cavi_q_μ = copy(q_μ) # zeros(P)
    cavi_q_var = copy(q_var) # ones(P) * 0.001
    # q_sd = sqrt.(q_var)
    cavi_q_α = copy(q_α) # ones(P) .* 0.10
    # nn_q_odds = ones(P) 
    # SSR = ones(P)

    X_sd = sqrt.(D ./ N)

    nn_p_causal = 0.01 * ones(P)
    nn_σ2_β = 0.0001 * ones(P)

    prev_loss = -Inf

    posterior_effect_sizes = []
    combined_cavi_losses = []
    corr_true_estimated = []

    for i in 1:max_iter
        println("Iteration $i")
        # train CAVI using set slab variance and p_causal as inputs; first round
        # cavi_q_u is cavi trained estimated betas, and coef is from iteration before
        q_μ, q_α, q_var, odds, new_loss, cavi_losses = train_cavi(cavi_q_μ, cavi_q_α, cavi_q_var, nn_p_causal, nn_σ2_β, X_sd, i, coef, SE, R, D)

        println("difference from n, n-1 (%)")
        println(abs(new_loss - prev_loss) / abs(prev_loss))

        # check for convergence
        if abs(new_loss - prev_loss) / abs(prev_loss) < threshold
            println("converged!")
            break
        end

        prev_loss = copy(new_loss)

        coef = cavi_q_μ .* cavi_q_α
        SE = sqrt.(cavi_q_var) 
        ## ak: Set α(i) =α and μ(i) =μ
        cavi_q_μ = copy(q_μ)
        cavi_q_α = copy(q_α)
        cavi_q_var = copy(q_var)

        max_abs_post_effect = abs(maximum(q_μ .* q_α))
        corr_with_true = cor(cavi_q_μ .* q_α, true_betas)

        # compute new marginal variance from q_α and q_var
        marg_var =  q_α .* q_var
        if any(q_var .< 0) | any(marg_var .< 0)
            error("q_var or marginal_var has neg value")
        end

        println("MARGINAL VARIANCE")
        println(marg_var)
        # marg_var = marg_var ./ mean(marg_var)

        println("Q_ALPHA")
        println(q_α)
        # train the neural network using G and the new s and p_causal
        model = fit_heritability_nn(marg_var, q_α, G, i) #*#

        # use the model to compute new σ2_β and p_causal
        outputs = model(transpose(G))
        println("VARIANCE B AFTER NN")
        println(outputs[1, :])
        # nn_σ2_β = expm1.(outputs[1, :]) ## ak: apply inverse of log1p to first output; cleaner way to write exp(x)-1
        nn_σ2_β = exp.(outputs[1, :]) 
        nn_p_causal = 1 ./ (1 .+ exp.(-outputs[2, :])) ## ak: logistic to recover orig prob
        println("new variance after training")
        println(nn_σ2_β)
        println("q odds after training")
        println(nn_p_causal)

        push!(posterior_effect_sizes, max_abs_post_effect)
        push!(corr_true_estimated, corr_with_true)
        push!(combined_cavi_losses, cavi_losses)

    end
    
    plot_max_effect = plot_max_effect_size_vs_iteration(posterior_effect_sizes)
    savefig("~/Downloads/scprs_figs/plot_max_effect.png")
    display(plot_max_effect)
    plot_corr = plot_corr_true_estimated(corr_true_estimated)
    savefig("~/Downloads/scprs_figs/plot_corr.png")
    display(plot_corr)
    println(posterior_effect_sizes)
    println("cavi losses")
    println(combined_cavi_losses)

    return q_μ, q_α, q_var
end
