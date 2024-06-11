"""
 fit_heritability_nn(model, q_μ, q_var, q_alpha, G)

 Fit the heritability neural network model.

    # Arguments
    - `model::Chain`: A neural network model
    - `q_μ::Vector`: A length P vector of posterior means
    - `q_var::Vector`: A length P vector of posterior variances
    - `q_α::Vector`: A length P vector of posterior probabilities of being causal
    - `G::AbstractArray`: A P x K matrix of annotations

```julia-repl
    model = Chain(
        Dense(20 => 5, relu; init = Flux.glorot_normal(gain = 0.0005)),
        Dense(5 => 2)
    )

    G = rand(Normal(0, 1), 100, 20)
    q_var = (G * rand(Normal(0, 0.10), 20)) .^ 2
    q_α = 1.0 ./ (1.0 .+ exp.(-1.0 .* (-2.0 .+ q_var)))
    trained_model = PRSFNN.fit_heritability_nn(
        model, 
        q_var, 
        q_α, 
        G
    )
    yhat = transpose(trained_model(transpose(G)))
    yhat[:, 1] .= exp.(yhat[:, 1])
    yhat[:, 2] .= 1.0 ./ (1.0 .+ exp.(-yhat[:, 2]))
```
"""
function fit_heritability_nn(model, q_var, q_α, G, i=1; max_epochs=3000, patience=400, mse_improvement_threshold=0.01, test_ratio=0.2, num_splits=10, optim_type = AdamW(0.02))

    # G_standardized = standardize(G)

    best_ks_statistic = Inf
    best_train_data = []
    best_test_data = []

    P = length(q_var)

    for _ in 1:num_splits
        # ak: shuffle indices
        permuted_indices = randperm(P)  
        # ak: get number of validation samples
        num_test = floor(Int, test_ratio * P)
        # ak: split into train and test based on above indices
        train_indices = permuted_indices[1:end-num_test]
        test_indices = permuted_indices[end-num_test+1:end]

        # ak: compute the KS statistic and pick the split with smallest KS stats
        ks_test = ApproximateTwoSampleKSTest(q_α[train_indices], q_α[test_indices])
        ks_n = ks_test.n_x*ks_test.n_y/(ks_test.n_x+ks_test.n_y)
        ks_statistic = (sqrt(ks_n)*ks_test.δ)

        if ks_statistic < best_ks_statistic
            best_ks_statistic = ks_statistic
            best_train_data = [Float32.(G[train_indices, :]), Float32.(q_var[train_indices]), Float32.(q_α[train_indices])]
            best_test_data = [Float32.(G[test_indices, :]), Float32.(q_var[test_indices]), Float32.(q_α[test_indices])]
        end

    end

    opt = Flux.setup(optim_type, model)
    X = Float32.(transpose(best_train_data[1]))
    Y = Float32.(transpose(hcat(log.(best_train_data[2]), logit.(best_train_data[3]))))
    data = (X, Y)

    # Main.@infiltrate
    DL = Flux.DataLoader(data, batchsize=10, shuffle=true, rng = Random.seed!(1))
    
    best_loss = Inf
    best_model = deepcopy(model)
    count_since_best = 0
    best_model_epoch = 1
    train_losses = Float64[]
    test_losses = Float64[]

    # check_no_nan(data[1])

    # @show model.layers[1].weight
    # @show model.layers[1].bias

    @inbounds for epoch in 1:max_epochs
        train!(nn_loss, model, DL, opt)
        train_loss = nn_loss(
                model, 
                Float32.(transpose(best_train_data[1])), 
                Float32.(transpose(hcat(log.(best_train_data[2]), logit.(best_train_data[3]))))
            )
        push!(train_losses, train_loss)

        # ak: validation loss
        test_loss = nn_loss(
                model,
                Float32.(transpose(best_test_data[1])),
                Float32.(transpose(hcat(log.(best_test_data[2]), logit.(best_test_data[3]))))
            )
        push!(test_losses, test_loss)

        mse_improvement = (test_loss - best_loss) / test_loss
        @info "$(ltime()) Epoch: $epoch, Train loss: $(round(train_loss, digits=2)), Test loss: $(round(test_loss, digits=2)), Relative change (ideally negative): $(round(mse_improvement; digits = 2))"

        # if improvement from prev iteration is greater than threshold
        if mse_improvement < -mse_improvement_threshold
            best_loss = test_loss
            count_since_best = 0
            # set current epoch as to when best model was observed
            best_model_epoch = epoch
            # and save current model as best model
            best_model = deepcopy(model)
        else
            count_since_best += 1
        end

        # If no improvement for "patience" epochs, stop training
        if count_since_best >= patience
            @info "$(ltime()) Early stopping after $epoch epochs."
            break
        end

    end

    @info "$(ltime()) Best model taken from epoch $best_model_epoch."

    return best_model
end

function find_max_activation(layer, K)
    # create a set of one-hot encoded input patterns; 100 vectors
    # only one feature is active at a time
    dummy_inputs = [Float32.(I == j ? 1 : 0 for I in 1:K) for j in 1:K]

    # extract activation values of the neuron for each input pattern
    activations = [layer(input) for input in dummy_inputs]
    # Identify the input pattern that resulted in the maximum activation
    max_activations = [argmax(vec(activation)) for activation in activations]

    return max_activations
end

function predict_with_nn(model, G)
    outputs = model(transpose(G))
    nn_σ2_β = exp.(outputs[1, :]) 
    nn_p_causal = 1 ./ (1 .+ exp.(-outputs[2, :])) ## ak: logistic to recover orig prob
    return nn_σ2_β, nn_p_causal
end

# RMSE
function nn_loss(model, G, y; weight_slab = 1.0, weight_causal = 1.0) ## ak: need two losses for slab variance and percent causal 

    yhat = model(G)
    loss_slab = @views Flux.mse(yhat[1, :], y[1, :])
    weighted_loss_slab = weight_slab * loss_slab
    loss_causal = @views Flux.mse(yhat[2, :], y[2, :])
    weighted_loss_causal = weight_causal * loss_causal
    total_loss = weighted_loss_slab + weighted_loss_causal ## ak: losses summed to form the total loss for training
    return total_loss
end

# function nn_loss(model, batch; weight_slab = 1.0, weight_causal = 1.0) ## ak: need two losses for slab variance and percent causal 

#     @show batch 
#     G = batch[1]
#     y = batch[2]

#     return nn_loss(model, G, y; weight_slab = weight_slab, weight_causal = weight_causal)
# end
