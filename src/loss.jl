
function read_plink_compute_cor()
    LD_block_bed = SnpArray(SnpArrays.datadir("test_data/1_4380811-5913893_base.bed"))
    LD_block_matrix = convert(Matrix{Int8}, LD_block_bed)
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
    Calculate the summary statistic RSS likelihood:

    rss(β, coef, SE, R)

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
Compute the joint log probability of the model

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

