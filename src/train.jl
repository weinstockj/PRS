function check_no_nan(data)

    if sum(isnan.(data[1])) > 0
        error("NaN detected.")
    end
    
    if sum(isnan.(data[2])) > 0
        error("NaN detected.")
    end

    if sum(isnan.(data[3])) > 0
        error("NaN detected.")
    end
end

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
function fit_heritability_nn(model, q_var, q_α, G, i=1; max_epochs = 50, patience=30, mse_improvement_threshold=0.01, test_ratio=0.2, num_splits=5, weight_slab=1, weight_causal=1)

    # RMSE
    function loss(model, x, y_slab, y_causal) ## ak: need two losses for slab variance and percent causal 
        yhat = model(transpose(x))
        #println("yhat in loss; slab, causal")
        #println(yhat[1, 1:5], yhat[2, 1:5])
        loss_slab = Flux.mse(yhat[1, :], y_slab)
        weighted_loss_slab = weight_slab * loss_slab
        loss_causal = Flux.mse(yhat[2, :], y_causal)
        weighted_loss_causal = weight_causal * loss_causal
	#println("loss_slab = $weighted_loss_slab")
        #println("loss_causal = $weighted_loss_causal")
        total_loss = weighted_loss_slab + weighted_loss_causal ## ak: losses summed to form the total loss for training
        # if !isfinite(total_loss)
        #     println("loss_slab = $loss_slab")
        #     println("loss_causal = $loss_causal")
        #     println("y_hat[2, :] = $(yhat[2, :])")
        #     println("y_causal = $(y_causal)")
        #     error("Total loss is not finite.")
        # end
        return total_loss
    end

    function clamp(x, ϵ = 1e-4)
        x = max.(min.(x, 1.0 - ϵ), ϵ) # to avoid Inf with logit transformation later
        return x
    end

    function logit(x)
        x = clamp(x)   
        return log.(x ./ (1 .- x))
    end
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

        # ak: get matching posterior variances in training and testing
        train_posterior_vars = q_var[train_indices]
        test_posterior_vars = q_var[test_indices]

        # ak: compute the KS statistic and pick the split with smallest KS stats
        ks_test = ApproximateTwoSampleKSTest(train_posterior_vars, test_posterior_vars)
        ks_n = ks_test.n_x*ks_test.n_y/(ks_test.n_x+ks_test.n_y)
        ks_statistic = (sqrt(ks_n)*ks_test.δ)


        if ks_statistic < best_ks_statistic
            best_ks_statistic = ks_statistic
            best_train_data = [Float32.(G[train_indices, :]), Float32.(q_var[train_indices]), Float32.(q_α[train_indices])]
            best_test_data = [Float32.(G[test_indices, :]), Float32.(q_var[test_indices]), Float32.(q_α[test_indices])]
        end

        # @info "$(ltime()) best KS statistic = $best_ks_statistic"
    end

    opt = Flux.setup(Momentum(), model)
    data = [ 
            (
            best_train_data[1], 
            log.(best_train_data[2]), # later apply inverse of exp.(x) 
            logit.(best_train_data[3]) # later apply logistic to recover orig prob
        
            )
    ]
    
    # println(data)
    
    best_loss = Inf
    best_model = deepcopy(model)
    count_since_best = 0
    best_model_epoch = 1
    train_losses = Float64[]
    test_losses = Float64[]

    for epoch in 1:max_epochs
        check_no_nan(data[1])
        train!(loss, model, data, opt)
	#println("just trained")
        #yhat_just_trained = model(transpose(G))
        #println("yhat just trained; slab, causal")
        #println(yhat_just_trained[1, 1:5], yhat_just_trained[2, 1:5])
        train_loss = loss(model, best_train_data[1], log.(best_train_data[2]), logit.(best_train_data[3]))
        push!(train_losses, train_loss)
	#println("computed train loss")
        # ak: validation loss
        test_loss = loss(model, best_test_data[1], log.(best_test_data[2]), logit.(best_test_data[3]))
	#println("computed test loss")
        #println("Test loss = $test_loss")
        #println("Train loss = $train_loss")
        push!(test_losses, test_loss)

        # check for improvement in loss
        mse_improvement = (test_loss - best_loss) / test_loss

        #println("MSE improvement = $mse_improvement")
        # if improvement from prev iteration is greater than threshold
        if mse_improvement < -mse_improvement_threshold
            best_loss = test_loss
            count_since_best = 0
            # set current epoch as to when best model was observed
            best_model_epoch = epoch
            # and save current model as best model
            best_model = deepcopy(model)
	    #println("in best model")
	    #yhat_current_best_model = best_model(transpose(G))
	    #println("yhat current best model; slab, causal")
            #println(yhat_current_best_model[1, 1:5], yhat_current_best_model[2, 1:5])
            #println("$(ltime()) NEW BEST MODEL at $best_model_epoch")
        else
            count_since_best += 1
        end

        # If no improvement for "patience" epochs, stop training
        if count_since_best >= patience
            @debug "$(ltime()) Early stopping after $epoch epochs."
            break
        end

    end

    @info "$(ltime()) Best model taken from epoch $best_model_epoch."

    return best_model
end

function train_cavi(p_causal, σ2_β, X_sd, i_iter, coef, SE, R, D, to; P = 1_000, n_elbo = 10, max_iter = 2, N = 10_000, σ2 = 1.0)
   
    function clamp_ssr(ssr, max_value = 709.7) # slightly below the threshold
        return min.(ssr, max_value)
    end

    @timeit to "initialize" begin
        @info "$(ltime()) Initializing CAVI..."
        P = length(coef)

        q_μ = zeros(P)
        q_var = ones(P) * 0.001
        q_sd = sqrt.(q_var)
        q_α = ones(P) .* 0.10
        q_odds = ones(P) 
        SSR = ones(P)

        Xty = @timeit to "copy Xty" copy(coef .* D)
        XtX = @timeit to "Create XtX" Diagonal(X_sd) * R * Diagonal(X_sd) .* N

        loss = -Inf
        prev_loss = -Inf
        prev_prev_loss = -Inf
        cavi_loss = Float32[]

        SR = SE .* R
        Σ = @timeit to "Σ" SR .* SE' 
        λ = 1e-8
	# if # of SNPs in LD block is less than 10K use poet_cov
	# if P < 10_000
	#     try
            	# Σ_reg = @timeit to "Σ_reg_poet_cov" PDMat(Hermitian(poet_cov(Σ; K = floor(Int64, P / 3), τ = .02, N = N) + λ * I))
            # catch e
                # @info "$(ltime()) poet_cov error; likely PosDefException"
	# 	println(e)
                # @info "$(ltime()) adjust negative eigenvalues by adding λ"
	# 	Σ_reg = @timeit to "Σ_reg_lambda_diagonal" PDMat(Hermitian(Σ + λ * I))
	#     end
            # else
        # # Σ_reg = @timeit to "Σ_reg" PDMat(Hermitian(poet_cov(Σ; K = 50, τ = .03)))
            # Σ_reg = @timeit to "Σ_reg_lambda_diagonal" PDMat(Hermitian(Σ + λ * I))
        # end
        Σ_reg = @timeit to "Σ_reg_lambda_diagonal" PDMat(Hermitian(Σ + λ * I))
	SRSinv = @timeit to "SRSinv" SR .* (1 ./ SE')
    end

    @inbounds for i in 1:max_iter

        # if clause just to monitor loss convergence
        if (mod(i, 1) == 0) | (i == 1)
            loss = 0.0
            @timeit to "elbo estimate" begin
                @inbounds for z in 1:n_elbo
                    z = rand(Normal(0, 1), P)
                    # loss_old = loss_old + elbo(z, q_μ, log.(q_var), coef, SE, R, σ2_β, p_causal, to)
                    loss = loss + elbo(z, q_μ, log.(q_var), coef, Σ_reg, SRSinv, σ2_β, p_causal, to)
                    # @info "$(ltime()) loss_new = $loss, loss_old = $loss_old"
                end
            end
         
            loss = loss / n_elbo

            if isnan(loss) == true
                error("NaN loss detected.")
                break
            end

            @info "$(ltime()) iteration $i, loss = $(round(loss; digits = 2)) (bigger numbers are better)"  
            # ak: stopping criterion for oscillation
            if (prev_loss > prev_prev_loss && loss < prev_loss) || 
                (prev_loss < prev_prev_loss && loss > prev_loss)
                # ak: we want to keep q_μ, q_α, q_var, q_odds from i-1 iteration
                @info "$(ltime()) Oscillation detected. Stopping at iteration $i."
                break
            end

            # ak: stopping criterion for insufficient improvement (10%)
            # ak: added a small constant to avoid division by zero
            relative_improvement = abs(loss - prev_loss) / (abs(prev_loss) + 1e-8)  
            if relative_improvement < 0.10
            # ak: we want to keep q_μ, q_α, q_var, q_odds from i-1 iteration
                @info "$(ltime()) Insufficient improvement. Stopping at iteration $i."
                break
            end

            @timeit to "push cavi loss" push!(cavi_loss, Float32(loss))

            # ak: update the previous losses for the next iteration
            prev_prev_loss = prev_loss
            prev_loss = loss
        end

        # println("q_μ, q_α, q_var, q_odds updates happening")

        @timeit to "update q_var" q_var .= σ2 ./ (diag(XtX) .+ 1 ./ σ2_β) ## ak: eq 8; \s^2_k; does not depend on alpha and mu from previous
        @timeit to "update q_μ" begin
            @inbounds for k in 1:P
                J = setdiff(1:P, k)
                q_μ[k] = (view(q_var, k) ./ σ2) .* (view(Xty, k) .- sum(view(XtX, k, J) .* view(q_α, J) .* view(q_μ, J))) ## ak: eq 9; update u_k
            end
        end
        @timeit to "update SSR" SSR .= q_μ .^ 2 ./ q_var
        @timeit to "clamp SSR"  SSR = clamp_ssr(SSR)
	@timeit to "update q_odds" q_odds .= (p_causal ./ (1 .- p_causal)) .* q_sd ./ sqrt.(σ2_β) .* exp.(SSR ./ 2.0) ## ak: eq 10; update a_k 
        @timeit to "update q_α" q_α .= q_odds ./ (1.0 .+ q_odds)

        # println("q_μ")
        # println(q_μ[1:3])

    end

    @info "$(ltime()) CAVI updates finished"

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
    # Identify the input pattern that resulted in the maximum activation
    max_activations = [argmax(vec(activation)) for activation in activations]

    return max_activations
end

"""
    `train_until_convergence(coef, SE, R, D, G; max_iter = 20, threshold = 0.1, N = 10_000)`

    # Arguments
    - `coef::Vector`: A length P vector of effect sizes
    - `SE::Vector`: A length P vector of standard errors
    - 'R::AbstractArray': A P x P correlation matrix
    - 'D::Vector': A length P vector of the sum of squared genotypes
    - 'G::AbstractArray': A P x K matrix of annotations
    
"""
function train_until_convergence(coef::Vector, SE::Vector, R::AbstractArray, D::Vector, G::AbstractArray; model = model, max_iter = 2, threshold = 0.1, train_nn = true, N = 10_000) 

    to = TimerOutput()
    ## initialize
    @timeit to "initialize" begin
        @info "$(ltime()) Initializing..."
        P = length(coef)
        q_μ = zeros(P)
        q_α = ones(P) .* 0.01
        L = sum(q_α)
        q_var = ones(P) * 0.001

        cavi_q_μ = copy(q_μ) # zeros(P)
        cavi_q_var = copy(q_var) # ones(P) * 0.001
        cavi_q_α = copy(q_α) # ones(P) .* 0.10

        X_sd = sqrt.(D ./ N)

        if train_nn 
            nn_p_causal = 0.01 * ones(P)
            nn_σ2_β = 0.001 * ones(P)
        else
            @info "$(ltime()) Resetting max_iter from $max_iter to 1 because the nn is frozen"
            nn_σ2_β, nn_p_causal = predict_with_nn(model, G)
            max_iter = 1
        end

        K = size(G, 2)
        P = size(G, 1)

    end

    prev_loss = -Inf
    model_init = deepcopy(model)
    prev_model = deepcopy(model)
    prev_prev_model = deepcopy(model)

    posterior_effect_sizes = Float32[]
    combined_cavi_losses = Vector{Float32}[]
    corr_true_estimated = Float32[]

    for i in 1:max_iter
        @info "$(ltime()) Training outer-loop iteration $i"
        # train CAVI using set slab variance and p_causal as inputs; first round
        # cavi_q_u is cavi trained estimated betas, and coef is from iteration before
        # q_μ, q_α, q_var, odds, new_loss, cavi_losses = train_cavi(cavi_q_μ, cavi_q_α, cavi_q_var, nn_p_causal, nn_σ2_β, X_sd, i, coef, SE, R, D)
        @timeit to "train_cavi" begin
            q_μ, q_α, q_var, odds, new_loss, cavi_losses = train_cavi(
                nn_p_causal, 
                nn_σ2_β, 
                X_sd, 
                i, 
                coef, 
                SE, 
                R, 
                D,
                to, # the timer function
		P = P,
                N = N
            )
            @info "$(ltime()) Training CAVI finished"
        end

        @timeit to "GC" begin
            GC.gc()
        end

        if i >= 2
            @debug "$(ltime()) difference from n, n-1 (%) = $(abs(new_loss - prev_loss) / abs(prev_loss))"
        end

        # check for convergence
        if abs(new_loss - prev_loss) / abs(prev_loss) < threshold
            @info "$(ltime()) converged!"
            break
        end

        # plot_cavi_losses(cavi_losses, i)
        # savefig("cavi_loss_iter$i.png")

        prev_loss = copy(new_loss)

        ## ak: Set α(i) =α and μ(i) =μ
        cavi_q_μ = copy(q_μ)
        cavi_q_α = copy(q_α)
        cavi_q_var = copy(q_var)

        max_abs_post_effect = abs(maximum(cavi_q_μ .* cavi_q_α))

        # compute new marginal variance from q_α and q_var
        marg_var =  cavi_q_α .* cavi_q_var #q_α .* q_var
        if any(cavi_q_var .< 0) | any(marg_var .< 0)
            error("q_var or marginal_var has neg value")
        end

        if train_nn 
            # train the neural network using G and the new s and p_causal
            @timeit to "fit_heritability_nn" begin
                model = fit_heritability_nn(model, q_var, q_α, G, i) #*#
                trained_model = deepcopy(model)
            end

            nn_σ2_β, nn_p_causal = predict_with_nn(trained_model, G)
            @debug "$(ltime()) mean nn_p_causal = $(mean(nn_p_causal))"
            nn_p_causal = nn_p_causal .* L ./ sum(nn_p_causal)

            @timeit to "deepcopys" begin
                push!(posterior_effect_sizes, max_abs_post_effect)
                push!(combined_cavi_losses, cavi_losses)
                prev_prev_model = deepcopy(prev_model) #at iter 1, initialized model
                prev_model = deepcopy(model) #at iter 1, trained nn model
            end
        end
    end
    
    show(to)

    if train_nn
        return cavi_q_μ, cavi_q_α, cavi_q_var, prev_model
    else
        return cavi_q_μ, cavi_q_α, cavi_q_var, model
    end
end

function predict_with_nn(model, G)
    outputs = model(transpose(G))
    nn_σ2_β = exp.(outputs[1, :]) 
    nn_p_causal = 1 ./ (1 .+ exp.(-outputs[2, :])) ## ak: logistic to recover orig prob
    return nn_σ2_β, nn_p_causal
end
