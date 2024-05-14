function train_cavi(p_causal, σ2_β, X_sd, i_iter, coef, SE, R, D, to; P = 1_000, n_elbo = 10, max_iter = 5, N = 10_000, σ2 = 1.0)
   
    @timeit to "initialize" begin
        @info "$(ltime()) Initializing CAVI..."
        cavi_iter = 0
        P = length(coef)

        q_μ = zeros(P)
        q_var = ones(P) * 0.001
        q_sd = sqrt.(q_var)
        q_α = ones(P) .* 0.10
        q_odds = ones(P)
        SSR = ones(P)

        q_μ_best = copy(q_μ)
        q_var_best = copy(q_var)
        q_α_best = copy(q_α)
        q_odds_best = copy(q_odds)

        Xty = @timeit to "copy Xty" copy(coef .* D)
        XtX = @timeit to "Create XtX" Diagonal(X_sd) * R * Diagonal(X_sd) .* N

        mean_loss = -Inf
        best_loss = -Inf

        se_loss = 0.0
        best_se_loss = 0.0
        cavi_loss = Float32[]

        SR = SE .* R
        Σ = @timeit to "Σ" SR .* SE'
        λ = 1e-8
        Σ_reg = @timeit to "Σ_reg_lambda_diagonal" PDMat(Hermitian(Σ + λ * I))
        SRSinv = @timeit to "SRSinv" SR .* (1 ./ SE')
    end

    @inbounds for i in 1:max_iter

        elbo_loss = Float32[]
        @timeit to "elbo estimate" begin
            @inbounds for z in 1:n_elbo
                z = rand(Normal(0, 1), P)
                push!(elbo_loss, elbo(z, q_μ, log.(q_var), coef, Σ_reg, SRSinv, σ2_β, p_causal, to))
            end
        end
     
        mean_loss = sum(elbo_loss) / n_elbo
        se_loss = std(elbo_loss) / sqrt(n_elbo)
        loss_lower_ci = mean_loss - 1.00 * se_loss # 65% CI
        loss_upper_ci = mean_loss + 1.00 * se_loss # 65% CI

        if (isnan(mean_loss) == true | isinf(mean_loss) == true)
            error("NaN loss detected.")
            break
        end

        # @info "$(ltime()) iteration $i, loss = $(round(loss; digits = 2)) (bigger numbers are better)"  
        @info "$(ltime()) iteration $i, ELBO CI = [$(round(loss_lower_ci; digits = 2)), $(round(loss_upper_ci; digits = 2))]  (bigger numbers are better)"  
        # ak: added a small constant to avoid division by zero
        if best_loss > loss_lower_ci 
            @info "$(ltime()) Insufficient ELBO improvement. Stopping at iteration $i and returning parameter estimates from iteration $(i-1)."
            break
        end

        q_μ_best = copy(q_μ)
        q_α_best = copy(q_α)
        q_var_best = copy(q_var)
        q_odds_best = copy(q_odds)
        best_loss = copy(mean_loss)
        best_se_loss = copy(se_loss)

        @timeit to "push cavi loss" push!(cavi_loss, Float32(mean_loss))

        # ak: update the previous losses for the next iteration

        @info "$(ltime()) CAVI updates at iteration $i"
        @timeit to "update q_var" q_var .= σ2 ./ (diag(XtX) .+ 1 ./ σ2_β) ## ak: eq 8; \s^2_k; does not depend on alpha and mu from previous
        @timeit to "update sd" q_sd .= sqrt.(q_var)
        @timeit to "update q_μ" begin
            @inbounds @fastmath for k in 1:P
                J = setdiff(1:P, k)
                q_μ[k] = (view(q_var, k) ./ σ2) .* (view(Xty, k) .- sum(view(XtX, k, J) .* view(q_α, J) .* view(q_μ, J))) ## ak: eq 9; update u_k
            end
        end
        @timeit to "update SSR" SSR .= q_μ .^ 2 ./ q_var
        @timeit to "clamp SSR" SSR = clamp_ssr(SSR)
        @timeit to "update q_odds" q_odds .= (p_causal ./ (1 .- p_causal)) .* q_sd ./ sqrt.(σ2_β) .* exp.(SSR ./ 2.0) ## ak: eq 10; update a_k 
        @timeit to "update q_α" q_α .= q_odds ./ (1.0 .+ q_odds)
    end

    @info "$(ltime()) CAVI updates finished"

    return q_μ_best, q_α_best, q_var_best, q_odds_best, best_loss, best_se_loss
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
function train_until_convergence(coef::Vector, SE::Vector, R::AbstractArray, D::Vector, G::AbstractArray; model = model, max_iter = 4, threshold = 0.2, train_nn = true, N = 10_000) 

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
            nn_σ2_β, nn_p_causal = predict_with_nn(model, Float32.(G))
            max_iter = 1
        end

        K = size(G, 2)
        P = size(G, 1)

    end

    prev_loss = -Inf
    model_init = deepcopy(model)
    prev_model = deepcopy(model)

    for i in 1:max_iter
        @info "$(ltime()) Training outer-loop iteration $i"
        # train CAVI using set slab variance and p_causal as inputs; first round
        # cavi_q_u is cavi trained estimated betas, and coef is from iteration before
        # q_μ, q_α, q_var, odds, new_loss, cavi_losses = train_cavi(cavi_q_μ, cavi_q_α, cavi_q_var, nn_p_causal, nn_σ2_β, X_sd, i, coef, SE, R, D)
        @timeit to "train_cavi" begin
           q_μ, q_α, q_var, odds, loss, loss_se = train_cavi(
                nn_p_causal, 
                nn_σ2_β, 
                X_sd, 
                i, 
                coef, 
                SE, 
                R, 
                D,
                to, # the timer function
                P=P,
                N=N
            )
            @info "$(ltime()) Training CAVI finished"
        end

        if i % 2 == 0
            @timeit to "GC" begin
                GC.gc()
            end
        end

        if i >= 2
            @info "$(ltime()) difference from n, n-1 (%) = $(round(100 * (loss - prev_loss) / abs(prev_loss); digits = 1))%"
        end

        # check for convergence
        @info "$(ltime()) ELBO = $(round(loss; digits = 2)), previous ELBO = $(round(prev_loss; digits = 1)), ELBO SE = $(round(loss_se; digits = 1)), threshold = $(round(threshold; digits = 2))"

        if (loss - prev_loss) / loss_se < threshold
            @info "$(ltime()) ELBO did not increase by the required amount; breaking now"
            break 
        end

        if abs(loss - prev_loss) / loss_se < threshold
            @info "$(ltime()) Converged"
            break
        end

        prev_loss = copy(loss)

        ## ak: Set α(i) =α and μ(i) =μ
        cavi_q_μ = copy(q_μ)
        cavi_q_α = copy(q_α)
        cavi_q_var = copy(q_var)

        if any(cavi_q_var .< 0)
            error("cavi_q_var has neg value")
        end

        if train_nn
            # train the neural network using G and the new s and p_causal
            @timeit to "fit_heritability_nn" begin
                model = fit_heritability_nn(model, q_var, q_α, G, i) #*#
                trained_model = deepcopy(model)
            end

            nn_σ2_β, nn_p_causal = predict_with_nn(trained_model, Float32.(G))
            @debug "$(ltime()) mean nn_p_causal = $(mean(nn_p_causal))"
            nn_p_causal = nn_p_causal .* L ./ sum(nn_p_causal)

            @timeit to "deepcopys" begin
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
