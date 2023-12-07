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
using StatsBase: sample
using HypothesisTests: ApproximateTwoSampleKSTest
using SnpArrays
# import AbstractDifferentiation as AD
# using Enzyme
# using ReverseDiff: GradientTape, gradient, gradient!, compile, GradientConfig, DiffResults

function read_plink_compute_cor()
    const LD_block_bed = SnpArray(SnpArrays.datadir("/home/akim126/data-abattle4/april/hi_julia/phase3/1_4380811-5913893_base.bed"))
    LD_block_matrix = convert(Matrix{Float64}, LD_block_bed)
    LD_block_R = cor(LD_block_matrix)
    return LD_block_R
end


# P is number of SNPs, K is number of annotations
# output is normalized variance per SNP e.g., s * h2 / P where h2 / P 
# is the average SNP and s is a small adjustment
function draw_slab_s_given_annotations(P; K = 100)
   
    K_half = K ÷ 2
   
    # G = rand(Normal(0, 1.0), P, K) # matrix of annotations - could be made more realistic?
    G_cont = rand(Normal(0, 1.0), P, K_half)  # half continuous annotations (normally distributed)
    G_binom = rand(Binomial(1, 0.5), P, K_half)  # half binomial annotations
    G = hcat(G_cont, G_binom)  # combine annotations
    # G[:,zero_col] .= 0
   
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
    
    σ2 = zeros(P) # P = 1000, K = 100
    for i in 1:P
        # define variance of SNPs as sum of all of the ϕ(G) + one interaction term between 
        # the first two annotations to throw in some complexity
        σ2[i] = exp(sum([ϕ[j](G[i, j]) for j in 1:K]) + ϕ[1](G[i, 1]) * ϕ[2](G[i, 2]))
        # exp(sum([raw[9][j](raw[6][1, j]) for j in 1:100]) + raw[9][1](raw[6][1, 1]) * raw[9][2](raw[6][1, 2]))
    end

    s = σ2 ./ mean(σ2) # normalize by average variance

    return s, G, choices, ϕ, σ2
end

function standardize(matrix)
    return (matrix .- mean(matrix, dims=1)) ./ std(matrix, dims=1)
end

function plot_loss_vs_epochs(train_losses, test_losses, i, epoch_model_taken)
    plot(1:length(train_losses), train_losses, xlabel="Epochs", ylabel="Loss", label="train", title="Loss vs. Epochs, iteration $i, best at $epoch_model_taken", reuse=false)
    plot!(1:length(test_losses), test_losses, lc=:orange, label="test")
    vline!([epoch_model_taken], label="epoch of best")
end

# how many epochs do we need?
# is momentum the right choice here as optimizer?
## ak: modified to have two outputs, and defined two losses: loss_slab, loss_causal
function fit_heritability_nn(model, q_μ, q_var, q_α, G, i; n_epochs = 200, patience=100, mse_improvement_threshold=0.1, test_ratio=0.2, num_splits=10, weight_slab=1, weight_causal=1)


    # RMSE
    function loss(model, x, y_slab, y_causal) ## ak: need two losses for slab variance and percent causal 
        yhat = model(transpose(x))
        loss_slab = Flux.mse(yhat[1, :], y_slab)
        weighted_loss_slab = weight_slab * loss_slab
        loss_causal = Flux.mse(yhat[2, :], y_causal)
        weighted_loss_causal = weight_causal * loss_causal
        return weighted_loss_slab + weighted_loss_causal ## ak: losses summed to form the total loss for training
    end

    logit(x) = log((x) ./ (1.0 .- x)) 
    # G_standardized = standardize(G)

    best_ks_statistic = Inf
    best_train_data = ()
    best_test_data = ()

    P = length(q_var)

    for _ in 1:num_splits
        # ak: shuffle indices
        permuted_indices = randperm(P)  
        # ak: get number of validation samples
        num_test = floor(Int, test_ratio * P)
        # ak: split into train and test based on above indices
        train_indices = permuted_indices[1:end-num_test]
        test_indices = permuted_indices[end-num_test+1:end]

        # ak: get matching posterior means in training and testing
        train_posterior_means = q_μ[train_indices]
        test_posterior_means = q_μ[test_indices]

        # ak: compute the KS statistic and pick the split with smallest KS stats
        ks_test = ApproximateTwoSampleKSTest(train_posterior_means, test_posterior_means)
        ks_n = ks_test.n_x*ks_test.n_y/(ks_test.n_x+ks_test.n_y)
        ks_statistic = (sqrt(ks_n)*ks_test.δ)

        if ks_statistic < best_ks_statistic
            best_ks_statistic = ks_statistic
            best_train_data = (G[train_indices, :], q_var[train_indices], q_α[train_indices])
            best_test_data = (G[test_indices, :], q_var[test_indices], q_α[test_indices])
        end
    end

    # Momentum()
    # RAdam()
    # AdaDelta()
    # AdaGrad(0.01)
    opt = Flux.setup(Momentum(), model)
    data = [
            (
            Float32.(best_train_data[1]), # to address Float64 -> Float32 Flux complaint
            log.(best_train_data[2]), # later apply inverse of exp.(x) .- 1.0
            logit.(best_train_data[3]) # later apply logistic to recover orig prob
        )
        ]
    
    best_loss = Inf
    best_model = deepcopy(model)
    count_since_best = 0
    best_model_epoch = Inf
    train_losses = []
    test_losses = []

    for epoch in 1:n_epochs
        train!(loss, model, data, opt)
        train_loss = loss(model, best_train_data[1], log.(best_train_data[2]), logit.(best_train_data[3]))
        push!(train_losses, train_loss)
        # ak: validation loss
        test_loss = loss(model, best_test_data[1], log.(best_test_data[2]), logit.(best_test_data[3]))
        println("Test loss = $test_loss")
        push!(test_losses, test_loss)

        # check for improvement in loss
        mse_improvement = best_loss - test_loss
        # if improvement from prev iteration is greater than threshold
        if mse_improvement > mse_improvement_threshold
            best_loss = test_loss
            count_since_best = 0
            # set current epoch as to when best model was observed
            best_model_epoch = epoch
            # and save current model as best model
            best_model = deepcopy(model)
            println("NEW BEST MODEL at $best_model_epoch")
        else
            count_since_best += 1
        end

        # If no improvement for "patience" epochs, stop training
        if count_since_best >= patience
            println("Early stopping after $epoch epochs.")
            break
        end

    end

    println("Best model taken from epoch $best_model_epoch.")

    # Plotting the loss
    plot_nn_loss = plot_loss_vs_epochs(train_losses, test_losses, i, best_model_epoch)
    savefig("epoch_losses_$i.png")
    
    return best_model
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
    # s, G, _ = draw_slab_s_given_annotations(P)
    s, G, function_choices, phi, sigma_squared = draw_slab_s_given_annotations(P)

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
    return X, β, Y, Σ, s, G, γ, function_choices, phi, sigma_squared
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

function train_cavi(p_causal, σ2_β, X_sd, i_iter, coef, SE, R, D; n_elbo = 50, max_iter = 10, N = 10_000, σ2 = 1.0)
   
    # TODO: eventually, replace fixed slab variance and inclusion probabilty with 
    # annotation informed prior
    P = length(coef)

    q_μ = zeros(P)
    q_var = ones(P) * 0.001
    q_sd = sqrt.(q_var)
    q_α = ones(P) .* 0.10
    q_odds = ones(P) 
    SSR = ones(P)

    # X_sd = sqrt.(D ./ N)
    Xty = copy(coef .* D)
    XtX = Diagonal(X_sd) * R * Diagonal(X_sd) .* N

    loss = -Inf
    prev_loss = -Inf
    prev_prev_loss = -Inf
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
                println("NaN loss detected.")
                break
            end

            println("i = $i, loss = $loss (bigger numbers are better)")   

            # ak: stopping criterion for oscillation
            if (prev_loss > prev_prev_loss && loss < prev_loss) || 
                (prev_loss < prev_prev_loss && loss > prev_loss)
                # ak: we want to keep q_μ, q_α, q_var, q_odds from i-1 iteration
                println("Oscillation detected. Stopping at iteration $i.")
                break
            end

            # ak: stopping criterion for insufficient improvement (10%)
            # ak: added a small constant to avoid division by zero
            relative_improvement = abs(loss - prev_loss) / (abs(prev_loss) + 1e-8)  
            if relative_improvement < 0.10
            # ak: we want to keep q_μ, q_α, q_var, q_odds from i-1 iteration
                println("Insufficient improvement. Stopping at iteration $i.")
                break
            end

            push!(cavi_loss, loss)

            # ak: update the previous losses for the next iteration
            prev_prev_loss = prev_loss
            prev_loss = loss
        end

        println("q_μ, q_α, q_var, q_odds updates happening")

        q_var .= σ2 ./ (diag(XtX) .+ 1 ./ σ2_β) ## ak: eq 8; \s^2_k; does not depend on alpha and mu from previous
        @inbounds for k in 1:P
            J = setdiff(1:P, k)
            q_μ[k] = (view(q_var, k) ./ σ2) .* (view(Xty, k) .- sum(view(XtX, k, J) .* view(q_α, J) .* view(q_μ, J))) ## ak: eq 9; update u_k
        end
        SSR .= q_μ .^ 2 ./ q_var
        q_odds .= (p_causal ./ (1 .- p_causal)) .* q_sd ./ sqrt.(σ2_β) .* exp.(SSR ./ 2.0) ## ak: eq 10; update a_k 
        q_α .= q_odds ./ (1.0 .+ q_odds)

        println("q_μ")
        println(q_μ[1:3])

    end

    println("CAVI updates finished!")

    # with probability q_alpha, additive effect beta is normal with mean q_mu and variance q_var
    return q_μ, q_α, q_var, q_odds, loss, cavi_loss
end

function plot_max_effect_size_vs_iteration(effect_sizes)
    plot(1:length(effect_sizes), effect_sizes, xlabel="Iteration", ylabel="Maximum absolute effect size", legend=false, title="Max abs effect size at each iteration", reuse=false)
end

function plot_cavi_losses(cavi_loss, i)
    plot(1:length(cavi_loss), cavi_loss, xlabel="iteration (inner loop)", ylabel="loss", legend=false, title="Iteration $i, CAVI loss", reuse=false)
end

function plot_corr_true_estimated(true_estimated_corr)
    plot(1:length(true_estimated_corr), true_estimated_corr, xlabel="Iteration", ylabel="Correlation (r)", legend=false, title="Correlation of true and est betas at each iteration", reuse=false)
end

function visualize_weights(model, i) #, annotation_names)
    # extract weights from the first layer
    layer1_weights = model.layers[1].weight
    # extract weights from the second layer
    layer2_weights = model.layers[2].weight
    # visualize the weights
    plot(heatmap(layer1_weights, c=cgrad([:blue, :white, :red])), heatmap(layer2_weights), size=(1000,800), 
    title="Weights for Neurons after $i NN training") #, [-Inf, 0, Inf]
end

function find_max_activation(layer, K)
    # create a set of one-hot encoded input patterns; 100 vectors
    # only one feature is active at a time
    dummy_inputs = [Float32.(I == j ? 1 : 0 for I in 1:K) for j in 1:K]

    # extract activation values of the neuron for each input pattern
    activations = [layer(input) for input in dummy_inputs]
    println("ACTIVATIONS")
    println(activations)

    # Identify the input pattern that resulted in the maximum activation
    max_activations = [argmax(vec(activation)) for activation in activations]

    return max_activations
end


# coef, SE, Z, cor(X), D
function train_until_convergence(coef, SE, R, D, G, true_betas, function_choices; max_iter = 20, threshold = 0.1, N = 10_000) # max_iter = 30, 

    ## initialize
    P = length(coef)
    q_μ = zeros(P)
    q_α = ones(P) .* 0.10
    L = sum(q_α)
    q_var = ones(P) * 0.001

    cavi_q_μ = copy(q_μ) # zeros(P)
    cavi_q_var = copy(q_var) # ones(P) * 0.001
    cavi_q_α = copy(q_α) # ones(P) .* 0.10

    X_sd = sqrt.(D ./ N)

    nn_p_causal = 0.01 * ones(P)
    nn_σ2_β = 0.0001 * ones(P)

    K = size(G, 2)
    P = size(G, 1)
    H = 5 #3 #10

    layer_1 = Dense(K => H, relu; init = Flux.glorot_normal(gain = 0.0005))
    layer_output = Dense(H => 2)
    model = Chain(
        layer_1, layer_output
    )

    max_activations = find_max_activation(layer_1, K)
    println("FIND MAX ACTIVATION")
    println(max_activations)

    annotations_initial = string.("annot", collect(1:K))
    annotations_layer1 = string.("annot", collect(1:H))

    visualize_weights(model, 0)
    savefig("initial_weights.png")

    prev_loss = -Inf
    model_init = deepcopy(model)
    prev_model = deepcopy(model)
    prev_prev_model = deepcopy(model)

    posterior_effect_sizes = Float32[]
    combined_cavi_losses = []
    corr_true_estimated = Float32[]

    for i in 1:max_iter
        println("Iteration $i")
        # train CAVI using set slab variance and p_causal as inputs; first round
        # cavi_q_u is cavi trained estimated betas, and coef is from iteration before
        # q_μ, q_α, q_var, odds, new_loss, cavi_losses = train_cavi(cavi_q_μ, cavi_q_α, cavi_q_var, nn_p_causal, nn_σ2_β, X_sd, i, coef, SE, R, D)
        q_μ, q_α, q_var, odds, new_loss, cavi_losses = train_cavi(
            nn_p_causal, 
            nn_σ2_β, 
            X_sd, 
            i, 
            coef, 
            SE, 
            R, 
            D
        )

        println("difference from n, n-1 (%)")
        println(abs(new_loss - prev_loss) / abs(prev_loss))

        # check for convergence
        if abs(new_loss - prev_loss) / abs(prev_loss) < threshold
            println("converged!")
            break
        end

        plot_cavi_losses(cavi_losses, i)
        savefig("cavi_loss_iter$i.png")

        prev_loss = copy(new_loss)

        ## ak: Set α(i) =α and μ(i) =μ
        cavi_q_μ = copy(q_μ)
        cavi_q_α = copy(q_α)
        cavi_q_var = copy(q_var)

        max_abs_post_effect = abs(maximum(cavi_q_μ .* cavi_q_α))
        corr_with_true = cor(cavi_q_μ .* cavi_q_α, true_betas)
        println("corr_with_true")
        println(corr_with_true)

        # compute new marginal variance from q_α and q_var
        marg_var =  cavi_q_α .* cavi_q_var #q_α .* q_var
        if any(cavi_q_var .< 0) | any(marg_var .< 0)
            error("q_var or marginal_var has neg value")
        end

        println("q_μ")
        println(cavi_q_μ[1:3])

        println("MARGINAL VARIANCE")
        println(marg_var[1:3])

        println("conditional VARIANCE")
        println(cavi_q_var[1:3])
        # marg_var = marg_var ./ mean(marg_var)

        println("Q_ALPHA")
        println(cavi_q_α[1:3])

        # println("printing weights before nn")
        # @show layer_output.weight
        # test_weight = layer_output.weight

        # train the neural network using G and the new s and p_causal
        model = fit_heritability_nn(model, q_μ, q_var, q_α, G, i) #*#

        trained_model = deepcopy(model)
        visualize_weights(trained_model, i)
        savefig("post_trainig_weights_iter$i.png")
        println("NN weights saved")

        # println("printing weights after nn")
        # @show layer_output.weight

        # use model to compute new σ2_β and p_causal to use in CAVI at next iteration
        outputs = trained_model(transpose(G))
        println("VARIANCE B AFTER NN")
        nn_σ2_β = exp.(outputs[1, :]) 
        println("sum(nn_σ2_β)")
        println(sum(nn_σ2_β))
        nn_σ2_β = nn_σ2_β .* 1.0 ./ sum(nn_σ2_β)
        println(nn_σ2_β[1:3])
        nn_p_causal = 1 ./ (1 .+ exp.(-outputs[2, :])) ## ak: logistic to recover orig prob
        println("new variance after training")
        println(nn_σ2_β[1:3])

        println("mean nn_p_causal")
        println(mean(nn_p_causal))
        nn_p_causal = nn_p_causal .* L ./ sum(nn_p_causal)
        println("new p_causal after training")
        println(nn_p_causal[1:3])

        push!(posterior_effect_sizes, max_abs_post_effect)
        push!(corr_true_estimated, corr_with_true)
        push!(combined_cavi_losses, cavi_losses)


        prev_prev_model = deepcopy(prev_model) #at iter 1, initialized model
        prev_model = deepcopy(model) #at iter 1, trained nn model

    end
    
    # plot_max_effect = plot_max_effect_size_vs_iteration(posterior_effect_sizes)
    # savefig("plot_max_effect.png")
    # display(plot_max_effect)
    println("corr with true effects")
    println(corr_true_estimated)
    plot_corr = plot_corr_true_estimated(corr_true_estimated)
    savefig("plot_corr.png")
    # display(plot_corr)
    println(posterior_effect_sizes)
    println("cavi losses")
    println(combined_cavi_losses)

    final_layer1_weights = prev_prev_model.layers[1].weight
    # final_layer1_weights = model.layers[1].weight
    importance_scores = sum(abs.(final_layer1_weights), dims=1)
    importance_scores_squared = sum(abs2.(final_layer1_weights), dims=1)
    plot(heatmap(reshape(function_choices, (1,100)), c=palette(:Blues_3,3), size=(690,50), legend=:left),
    bar(1:length(vec(importance_scores)), vec(importance_scores), ylabel="abs(weight)", legend=false),
    bar(1:length(vec(importance_scores_squared)), vec(importance_scores_squared), ylabel="abs2(weight)", legend=false, c=:green),
    layout=(3,1), size=(700,600))
    xlabel!("annotations")
    savefig("learned_weights_score.png")

    function cavi_prior(beta, max_int)
        return cavi_q_α[max_int] * pdf(Normal(0, sqrt(cavi_q_var[max_int])), beta)
    end

    function cavi_prior_same_snp(beta, max_int)
        return cavi_q_α[max_int] * pdf(Normal(0, sqrt(cavi_q_var[max_int])), beta)
    end

    function original_prior(beta, pi)
        # ak: update later for no hardcoded values
        sigma_spike = sqrt(0.001)
        sigma_slab = sqrt(0.10 / 89)
    
        return pi * pdf(Normal(0, sigma_slab), beta) + (1 - pi) * pdf(Normal(0, sigma_spike), beta)
    end

    beta_grid = range(-0.1, stop=0.1, length=20)

    max_α = findmax(cavi_q_α)[2]
    # ak: update later for no hardcoded values
    Z = [original_prior(b1, 0.089) * cavi_prior(b2, max_α) for b1 in beta_grid, b2 in beta_grid]

    contour(beta_grid, beta_grid, Z, xlabel="Prior density, no training", ylabel="Prior density, NN training", color=:rust)
    plot!(beta_grid, beta_grid, label="x=y")
    plot!(size=(700,600))
    savefig("prior_density_at_max_q_alpha.png")

    # Z_same = [original_prior(b1, 0.089) * cavi_prior(b2, 517) for b1 in beta_grid, b2 in beta_grid]


    # contour(beta_grid, beta_grid, Z, xlabel="Prior density, no training", ylabel="Prior density, NN training", color=:rust)
    # plot!(beta_grid, beta_grid, label="x=y")
    # plot!(size=(700,600))
    # savefig("varying_G_density.png")

    # contour(beta_grid, beta_grid, Z_same, xlabel="Prior density, no training", ylabel="Prior density, NN training", color=:rust)
    # plot!(beta_grid, beta_grid, label="x=y")
    # plot!(size=(700,600))
    # savefig("varying_G_density_same.png")

    return cavi_q_μ, cavi_q_α, cavi_q_var, prev_prev_model, model_init #model
end
