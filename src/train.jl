"""
    train_cavi(p_causal, σ2_β, coef, SE, R, XtX, Xty, to; P = 1_000, n_elbo = 10, max_iter = 5, N = 10_000, yty = 10_000, spike_σ2 = 1e-6, update_σ2 = true, σ2 = 1.0)

Performs Coordinate Ascent Variational Inference (CAVI) for a spike-and-slab model to estimate
genetic variant effect sizes and posterior inclusion probabilities.

# Arguments
- `p_causal::Vector{Float64}`: Prior probabilities for variants to be causal
- `σ2_β::Vector{Float64}`: Prior variances for effect sizes (slab component)
- `coef::Vector{Float64}`: Effect sizes from GWAS summary statistics
- `SE::Vector{Float64}`: Standard errors of the effect sizes
- `R::Matrix{Float64}`: LD correlation matrix
- `XtX::Matrix{Float64}`: Matrix equal to N times the covariance matrix of genotypes
- `Xty::Vector{Float64}`: Vector of inner products between genotypes and phenotype
- `to::TimerOutput`: Timer output object for performance profiling

# Keyword Arguments
- `P::Int64=1_000`: Number of variants
- `n_elbo::Int64=10`: Number of Monte Carlo samples for ELBO estimation
- `max_iter::Int64=5`: Maximum number of iterations for convergence
- `N::Float64=10_000`: Sample size
- `yty::Float64=10_000`: Sum of squared phenotypes
- `spike_σ2::Float64=1e-6`: Variance for the spike component
- `update_σ2::Bool=true`: Whether to update the residual variance
- `σ2::Float64=1.0`: Initial residual variance

# Returns
A named tuple containing:
- `q_μ::Vector{Float64}`: Posterior means for the slab component
- `q_spike_μ::Vector{Float64}`: Posterior means for the spike component
- `q_α::Vector{Float64}`: Posterior inclusion probabilities
- `q_var::Vector{Float64}`: Posterior variances for the slab component
- `q_odds::Vector{Float64}`: Posterior odds for variant inclusion
- `loss::Float64`: Final ELBO value
- `se_loss::Float64`: Standard error of the ELBO estimate
- `σ2::Float64`: Final residual variance estimate

# Details
This function implements an iterative Coordinate Ascent Variational Inference algorithm for a 
spike-and-slab model, commonly used in Polygenic Risk Score (PRS) calculation. At each iteration,
it updates the variational parameters to maximize the Evidence Lower Bound (ELBO).

The algorithm stops when insufficient improvement in ELBO is detected or when the maximum
number of iterations is reached. The best parameters (with highest ELBO) are returned.
"""
function train_cavi(p_causal, σ2_β, coef, SE, R, XtX, Xty, to; P = 1_000, n_elbo = 10, max_iter = 5, N = 10_000, yty = 10_000, spike_σ2 = 1e-6, update_σ2 = true, σ2 = 1.0) #spike_σ2 = 1e-5
   
    @timeit to "initialize" begin
        @info "$(ltime()) Initializing CAVI..."
        cavi_iter = 0
        P = length(coef)

        q_μ = zeros(P)
        q_spike_μ = zeros(P)
        q_var = ones(P) * 0.001
        q_spike_var = ones(P) * 0.001
        q_sd = sqrt.(q_var)
        q_spike_sd = sqrt.(q_spike_var)

        q_α = ones(P) .* 0.10
        q_odds = ones(P)
        SSR = ones(P)
        if update_σ2
  	        σ2 = 1.0
        end
	end

    q_μ_best = copy(q_μ)
    q_spike_μ_best = copy(q_spike_μ)
    q_var_best = copy(q_var)
    q_spike_var_best = copy(q_spike_var)
    q_α_best = copy(q_α)
    q_odds_best = copy(q_odds)
    σ2_best = σ2

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

    @inbounds for i in 1:max_iter

        elbo_loss = Float32[]
        @timeit to "elbo estimate" begin
            @info "$(ltime()) current σ2 for elbo estimate = $(σ2)"
            @inbounds for z in 1:n_elbo
                z = rand(Normal(0, 1), P)
                push!(elbo_loss, elbo(z, q_μ, log.(q_var), coef, Σ_reg, SRSinv, σ2_β, p_causal, σ2, spike_σ2, to))
            end
        end
     
        mean_loss = sum(elbo_loss) / n_elbo
        se_loss = std(elbo_loss) / sqrt(n_elbo)
        loss_lower_ci = mean_loss - 1.00 * se_loss # 65% CI
        loss_upper_ci = mean_loss + 1.00 * se_loss # 65% CI

        if (isnan(mean_loss) | isinf(mean_loss))
            error("NaN loss detected.")
            break
        end

        @info "$(ltime()) iteration $i, ELBO CI = [$(round(loss_lower_ci; digits = 2)), $(round(loss_upper_ci; digits = 2))]  (bigger numbers are better)" 

        if best_loss > loss_lower_ci 
            @info "$(ltime()) Insufficient ELBO improvement. Stopping at iteration $i and returning parameter estimates from iteration $(i-1)."
            break
        end

        q_μ_best = copy(q_μ)
        q_spike_μ_best = copy(q_spike_μ)
        q_α_best = copy(q_α)
        q_var_best = copy(q_var)
        q_spike_var_best = copy(q_spike_var)
        q_odds_best = copy(q_odds)
        best_loss = copy(mean_loss)
        best_se_loss = copy(se_loss)
	    σ2_best = σ2

        @timeit to "push cavi loss" push!(cavi_loss, Float32(mean_loss))

        @info "$(ltime()) CAVI updates at iteration $i"
        @timeit to "update q_var" q_var .= σ2 ./ (diag(XtX) .+ 1 ./ σ2_β) ## ak: eq 8; \s^2_k; does not depend on alpha and mu from previous
        @timeit to "update q_spike_var" q_spike_var .= σ2 ./ (diag(XtX) .+ 1 ./ spike_σ2) ## ak: eq 8; \s^2_k; does not depend on alpha and mu from previous
        @timeit to "update q_sd" q_sd .= sqrt.(q_var)
        @timeit to "update q_spike_sd" q_spike_sd .= sqrt.(q_spike_var)
        @timeit to "update q_μ" begin
            # inner_loop_cavi!(q_μ, q_spike_μ, q_α, q_var, q_spike_var, XtX, Xty, σ2; P = P)
            inner_loop_cavi_fast!(q_μ, q_spike_μ, q_α, q_var, q_spike_var, XtX, Xty, σ2; P = P)
        end
        @timeit to "update SSR" SSR .= q_μ .^ 2 ./ q_var
        @timeit to "clamp SSR" SSR = clamp_ssr(SSR)
        @timeit to "update q_odds" q_odds .= (p_causal ./ (1 .- p_causal)) .* q_sd ./ sqrt.(σ2_β .* σ2) .* exp.(SSR ./ 2.0) 
        @timeit to "update q_α" q_α .= q_odds ./ (1.0 .+ q_odds)

        if update_σ2
                @timeit to "update σ2" begin
                @info "$(ltime()) σ2 is being updated."
                    a = (1 + median(N) + P) / 2 
                    b = (1.0 + yty - 2 * sum(q_μ .* Xty) + q_μ' * XtX * q_μ) / 2
                    if b < 0
                        # Main.@infiltrate
                        error("Negative shape parameter 'b' in InverseGamma distribution: $b.")
                    end
                    error_dist = InverseGamma(a, b)
                    σ2 = mean(error_dist)
                    @info "$(ltime()) σ2 estimate = $(round(σ2; digits = 3))"
                end
        end
    end

    @info "$(ltime()) CAVI updates finished"

    return (
        q_μ = q_μ_best, 
        q_spike_μ = q_spike_μ_best, 
        q_α = q_α_best, 
        q_var = q_var_best, 
        q_odds = q_odds_best, 
        loss = best_loss, 
        se_loss = best_se_loss, 
        σ2 =  σ2_best
    )
end

function inner_loop_cavi!(q_μ, q_spike_μ, q_α, q_var, q_spike_var, XtX, Xty, σ2; P = 1_000)
    @inbounds @fastmath for k in 1:P

        J = setdiff(1:P, k)

        q_μ[k] = (view(q_var, k) ./ σ2) .* 
        (view(Xty, k) .- sum(view(XtX, k, J) .* (view(q_α, J) .* view(q_μ, J) .+ view(1.0 .- q_α, J) .* view(q_spike_μ, J))))
        
        q_spike_μ[k] = (view(q_spike_var, k) ./ σ2) .* 
        (view(Xty, k) .- sum(view(XtX, k, J) .* (view(q_α, J) .* view(q_μ, J) .+ view(1.0 .- q_α, J) .* view(q_spike_μ, J)))) 
    end
end

function inner_loop_cavi_fast!(q_μ, q_spike_μ, q_α, q_var, q_spike_var, XtX, Xty, σ2; P = 1_000)
    
    marginal_posterior_mean = q_μ .* q_α .+ q_spike_μ .* (1 .- q_α)
    
    @inbounds @fastmath for k in 1:P

        inner_term = @views sum(XtX[:, k] .* marginal_posterior_mean)

        q_μ[k] = (q_var[k] ./ σ2) .* 
            (Xty[k] .- (inner_term - XtX[k, k] * marginal_posterior_mean[k]))

        q_spike_μ[k] = (q_spike_var[k] ./ σ2) .* 
            (Xty[k] .- (inner_term - XtX[k, k] * marginal_posterior_mean[k]))

        marginal_posterior_mean[k] = q_μ[k] * q_α[k] + q_spike_μ[k] * (1 - q_α[k])

    end
end

"""
    infer_σ2(coef, SE, R, D, X_sd, N, P)

# Arguments
- `coef::Vector`: A length P vector of effect sizes
- `SE::Vector`: A length P vector of standard errors
- `XtX::AbstractArray`: A P x P matrix equal to N times the covariance matrix of the genotypes
- `Xty::Vector`: A length P vector of the inner product between genotype and phenotype
- `N`: Number of samples
- `P`: Number of SNPs
"""
function infer_σ2(coef::Vector, SE::Vector, XtX::AbstractArray, Xty::Vector, N::Real, P::Int64; estimate = false, λ = 100)

    D = construct_D(XtX)

    if estimate
        prob = LinearProblem(XtX + λ * I, Xty)
        init(prob);
        sol = solve(prob)
        β_joint = sol.u
        yty = median(D .* (SE .^ 2) .* (N - 1) .+ D .* (coef .^ 2)) 
        R2 = β_joint' * Xty / yty
    else
        R2 = 0.0 # assume no h2
    end
    yty = median(D .* (SE .^ 2) .* (N - 1) .+ D .* (coef .^ 2)) 
    σ2 = (1 - R2) * yty / (N - P)

    GC.gc()

    @assert σ2 > 0 "σ2 is negative"

    return (; σ2, R2, yty)
end

"""
    train_until_convergence(coef, SE, R, D, G; max_iter = 20, threshold = 0.1, N = 10_000)

# Arguments
- `coef::Vector`: A length P vector of effect sizes
- `SE::Vector`: A length P vector of standard errors
- `R::AbstractArray`: A P x P correlation matrix
- `D::Vector`: A length P vector of the sum of squared genotypes
- `G::AbstractArray`: A P x K matrix of annotations
    
"""
function train_until_convergence(coef::Vector, SE::Vector, R::AbstractArray, XtX::AbstractArray, Xty::Vector, G::AbstractArray; model = model, opt = opt, max_iter = 4, threshold = 0.2, train_nn = true, N = 10_000, yty = 300_000, σ2 = 1.0, R2 = 0.01, update_σ2 = true) 

    to = TimerOutput()
    ## initialize
    @timeit to "initialize" begin
        @info "$(ltime()) Initializing..."
        P = length(coef)
        q_μ = zeros(P)
        q_spike_μ = zeros(P)
        q_α = ones(P) .* 0.01
        # L = sum(q_α)
        q_var = ones(P) * 0.001
	    updated_σ2 = σ2

        cavi_q_μ = copy(q_μ) # zeros(P)
        cavi_q_spike_μ = copy(q_spike_μ) # zeros(P)
        cavi_q_var = copy(q_var) # ones(P) * 0.001
        cavi_q_α = copy(q_α) # ones(P) .* 0.10
	    cavi_updated_σ2 = updated_σ2

        @info "$(ltime()) Estimated σ2 = $(round(σ2; digits = 2)), h2 = $(round(R2; digits = 2))"

        if (model != nothing) & !train_nn
            @info "$(ltime()) Resetting max_iter from $max_iter to 1 because the nn is frozen"
            @info "$(ltime()) Confirming global residual variance $(round(σ2; digits = 2))."
	        nn_σ2_β, nn_p_causal = predict_with_nn(model, Float32.(G))
            nn_σ2_β = nn_σ2_β .* σ2
            max_iter = 1
	        update_σ2 = false
        else
            @info "$(ltime()) Initializing prior inclusion probability and slab variance"
            nn_p_causal = 0.1 .* ones(P)
            nn_σ2_β = σ2 .* 0.001 .* ones(P)
        end

        if !train_nn 
            max_iter = 1
            @info "$(ltime()) Resetting max_iter from $max_iter to 1 because the nn is frozen"
        end

        K = size(G, 2)
        P = size(G, 1)

    end

    prev_loss = -Inf
    model_init = deepcopy(model)
    prev_model = deepcopy(model)

    for i in 1:max_iter
        @info "$(ltime()) Training outer-loop iteration $i"
        @timeit to "train_cavi" begin
            q_μ, q_spike_μ, q_α, q_var, odds, loss, loss_se, updated_σ2 = train_cavi(
                nn_p_causal, 
                nn_σ2_β, 
                coef, 
                SE, 
                R, 
                XtX,
                Xty,
                to; # the timer function
                P = P,
                N = N,
                yty = yty,
		        update_σ2 = update_σ2,
		        σ2 = σ2
            )
            @info "$(ltime()) Training CAVI finished"
        end

        if i % 2 == 0
            @timeit to "GC" begin
                GC.gc()
            end
        end

        if i >= 2
            @info "$(ltime()) difference from n, n-1 (%) = $(round(100 * (loss - prev_loss) / abs(prev_loss); digits = 2))%"
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

        cavi_q_μ = copy(q_μ)
        cavi_q_spike_μ = copy(q_spike_μ)
        cavi_q_α = copy(q_α)
        cavi_q_var = copy(q_var)
        # cavi_q_marginal_var = compute_marginal_variance(q_μ, q_var, q_α)
	    cavi_updated_σ2 = updated_σ2

        @info "q_μ / sqrt(σ2)"
        describe_vector(q_μ ./ sqrt(σ2))

        @info "q_μ ^ 2 / σ2"
        describe_vector(q_μ .^ 2 / σ2)

        @info "$(ltime()) sum(q_α) = $(round(sum(q_α), digits = 2)), sum(cavi_q_marginal_var) = $(round(sum(q_var), digits = 2)), std(q_μ) = $(round(std(q_μ), digits = 2))"
        @info "$(ltime()) Inferred $(round(sum(cavi_q_α .> .50), digits = 2)) variants with PIP >= 50%"

        if train_nn 
            # train the neural network using G and the new s and p_causal
            @timeit to "fit_heritability_nn" begin
                # model = fit_heritability_nn(model, q_var, q_α, G, i) #*#
                model = fit_heritability_nn(model, opt, q_μ .^ 2 / σ2, q_α, G, i) #*#
                trained_model = deepcopy(model)
            end

            nn_σ2_β, nn_p_causal = predict_with_nn(trained_model, Float32.(G))

            @info "nn_p_causal before normalization"
            describe_vector(nn_p_causal)

            @info "nn_σ2_β before normalization"
            describe_vector(nn_σ2_β)


            @timeit to "deepcopys" begin
                prev_model = deepcopy(model) #at iter 1, trained nn model
            end
        end
    end


    @info "$(ltime()) Inferred $(round(sum(cavi_q_α .> .50), digits = 2)) variants with PIP >= 50%"

    @info "$(ltime()) Training finished"

    show(to)

    println("\n")

    if train_nn
        return (
            q_μ = cavi_q_μ, 
            q_α = cavi_q_α, 
            q_spike_μ = cavi_q_spike_μ, 
            nn_σ2_β = nn_σ2_β, 
            nn_p_causal = nn_p_causal, 
            cavi_updated_σ2 = cavi_updated_σ2,
            prev_model = prev_model
        )
    else
        return (
            q_μ = cavi_q_μ, 
            q_α = cavi_q_α, 
            q_spike_μ = cavi_q_spike_μ, 
            nn_σ2_β = nn_σ2_β, 
            nn_p_causal = nn_p_causal, 
            cavi_updated_σ2 = cavi_updated_σ2
        )
    end
end

function describe_vector(x::Vector, digits = 4)
    @info "mean = $(round(mean(x); digits = digits)), std = $(round(std(x); digits = digits)), min = $(round(minimum(x); digits = digits)), max = $(round(maximum(x); digits = digits)), sum = $(round(sum(x); digits = digits))"
end

function compute_marginal_variance(q_μ, q_var, p_slab = 0.01, spike_σ2 = 1e-8)
    return p_slab .* q_var .+ (1 .- p_slab) .* spike_σ2 .+ (p_slab .* q_μ .^ 2 .- (p_slab .* q_μ) .^ 2)
end

function train_gibbs(p_causal, σ2_β, coef, SE, R, XtX, Xty, to; P = 1_000, max_iter = 150, N = 10_000, yty = 10_000, spike_σ2 = 1e-5, λ = 10_000)

    warmup = 50
    thin = 2
    ## initialize
    σ2 = 1.0
    β_draw = zeros(P, max_iter)
    slab_prob = zeros(P)
    spike_prob = zeros(P)
    γt = zeros(P, max_iter)
    γt[:, 1] .= rand.(Bernoulli.(ones(P) .* 0.5))
    σ2t = ones(max_iter)
    σ2t[1] = σ2
    slab_dist = Normal.(0, sqrt(σ2t[1]) .* sqrt.(σ2_β .+ spike_σ2))
    α = zeros(P)

    Dt = PDiagMat(zeros(P))
    Σt = PDMat(Symmetric(XtX + λ * I))
    Σtinv = zeros(P, P)

    @info "$(ltime()) Starting Gibbs sampling..."
    for t in 2:max_iter
        if t % 50 == 0
            @info "$(ltime()) Gibbs sampling iteration $t out of $max_iter"
        end
        @timeit to "gibbs" begin
            @timeit to "update Dt" Dt = @views PDiagMat(γt[:, t - 1] ./ σ2_β .+ (1.0 .- γt[:, t - 1]) ./ spike_σ2)
            @timeit to "update Σt" Σt = PDMat(XtX + Dt + λ * I)
            # @timeit to "update Σt" pdadd!(Σt, Dt)

            # @timeit to "update Σt" Σt .+= Dt
            @timeit to "update Σtinv" Σtinv .= inv(Σt)
            @timeit to "define β_dist" β_dist = @views MvNormal(Σtinv * Xty, Σtinv * σ2t[t - 1])
            # Main.@infiltrate
            @timeit to "draw β" β_draw[:, t] .= rand(β_dist)
            @timeit to "define slab dist" slab_dist .= @views Normal.(0, sqrt(σ2t[t - 1]) .* sqrt.(σ2_β .+ spike_σ2))
            @timeit to "define spike dist" spike_dist = @views Normal(0, sqrt(σ2t[t - 1]) .* sqrt(spike_σ2))
            @timeit to "compute PIP" begin
                slab_prob .= @views pdf.(slab_dist, β_draw[:, t])
                spike_prob .= @views pdf(spike_dist, β_draw[:, t])
                α .= (p_causal .* slab_prob) ./ (p_causal .* slab_prob .+ (1 .- p_causal) .* spike_prob)
            end
            # if t > 10
            #     Main.@infiltrate
            # end
            @timeit to "draw γ" γt[:, t] .= rand.(Bernoulli.(α))
            @timeit to "update σ2" begin
                a = (1 + median(N) + P) / 2 
                b = @views (1.0 + yty - 2 * sum(β_draw[:, t] .* Xty) + β_draw[:, t]' * XtX * β_draw[:, t] + β_draw[:, t]' * Dt * β_draw[:, t]) / 2
                # b = @views (1.0 + yty - 2 * sum(β_draw[:, t] .* Xty) + quad(XtX, β_draw[:, t]) + β_draw[:, t]' * Dt * β_draw[:, t]) / 2
                error_dist = InverseGamma(a, b)
                σ2t[t] = rand(error_dist)
            end
            # @timeit to "update Σt" Σt .-= Dt
        end
    end

    # return vec(mean(β_draw[:, warmup:thin:max_iter], dims = 2)), vec(mean(γt[:, warmup:thin:max_iter], dims = 2)), α, mean(σ2t[warmup:thin:max_iter])
    return vec(mean(β_draw[:, warmup:thin:max_iter], dims = 2)), vec(mean(γt[:, warmup:thin:max_iter], dims = 2)), α
end

