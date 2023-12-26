# P is number of SNPs, K is number of annotations
# output is normalized variance per SNP e.g., s * h2 / P where h2 / P 
# is the average SNP and s is a small adjustment
function draw_slab_s_given_annotations(P; K = 100)
   
    K_half = K ÷ 2
   
    # G = rand(Normal(0, 1.0), P, K) # matrix of annotations - could be made more realistic?
    G_cont = rand(Normal(0, 1.0), P, K_half)  # half continuous annotations (normally distributed)
    G_binom = rand(Binomial(1, 0.5), P, K_half)  # half binomial annotations
    G::Matrix{Float64} = hcat(G_cont, G_binom)  # combine annotations
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
    ϕ::Vector{Function} = [functions[choices[i]] for i in 1:K] # randomly pick a linear or quadratic transformation
    
    σ2 = zeros(P) # P = 1000, K = 100
    @inbounds for i in 1:P
        # define variance of SNPs as sum of all of the ϕ(G)
        @inbounds for j in 1:K
            σ2[i] += ϕ[j](G[i, j])
        end
        # σ2[i] = exp(σ2[i] + ϕ[1](G[i, 1]) * ϕ[2](G[i, 2]))
        σ2[i] = exp(σ2[i])
        # exp(sum([raw[9][j](raw[6][1, j]) for j in 1:100]) + raw[9][1](raw[6][1, 1]) * raw[9][2](raw[6][1, 2]))
    end

    s::Vector{Float64} = σ2 ./ mean(σ2) # normalize by average variance

    return s, G, choices, ϕ, σ2
end

function simulate_raw(;N = 10_000, P = 1_000, K = 100)

    Random.seed!(0)

    eigenvalues = sort(rand(Exponential(1), P))
    eigenvalues[length(eigenvalues)] = eigenvalues[length(eigenvalues)] * 10
    D = Diagonal(eigenvalues)
    U = rand(Normal(0, 1), P, P)
    U, _ = qr(U)
    Σ = U * D * U'
    Σ = 0.5 * (Σ + Σ')
    X = transpose(rand(MvNormal(zeros(P), Σ), N))
    p_causal = 0.10
    L = p_causal * P # 100
    γ = rand(Bernoulli(p_causal), P)
    h2 = 0.10

    # Simulation annotations and create expected s parameter for causal SNPs
    # s, G, _ = draw_slab_s_given_annotations(sum(γ))
    # s, G, _ = draw_slab_s_given_annotations(P)
    s, G, function_choices, phi, sigma_squared = draw_slab_s_given_annotations(P; K = K)

    spike = rand(Normal(0, 0.001), P)
    slab  = rand(Normal(0,  sqrt(h2 / L)), P)
    β = γ .* slab + (1 .- γ) .* spike
    # β[γ] .= β[γ] .* sqrt.(s)
    β .= β .* sqrt.(s) ## ak: s is normalized variance per SNP for ALL SNPs, not just causal #*#
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

"""
`estimate_sufficient_statistics(X, Y)`

Estimate the sufficient statistics for the model given genotypes and phenotype

"""
function estimate_sufficient_statistics(X::AbstractArray, Y::Vector)
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
