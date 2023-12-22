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
                error("NaN loss detected.")
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

        # println("q_μ")
        # println(q_μ[1:3])

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
    # println("ACTIVATIONS")
    # println(activations)

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
