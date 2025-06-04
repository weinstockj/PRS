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

"""
    simulate_raw(; N = 10_000, P = 1_000, K = 100, h2 = 0.10)

Simulate genotype and phenotype data with genetic architecture influenced by functional annotations.

# Arguments
- `N::Int=10_000`: Number of samples/individuals
- `P::Int=1_000`: Number of genetic variants/SNPs
- `K::Int=100`: Number of functional annotations
- `h2::Float64=0.10`: Desired narrow-sense heritability of the trait

# Returns
A named tuple containing:
- `X::Matrix{Float64}`: Genotype matrix (N × P)
- `β::Vector{Float64}`: True causal effect sizes
- `Y::Vector{Float64}`: Simulated phenotypes
- `Σ::Matrix{Float64}`: Covariance matrix of genotypes (LD structure)
- `s::Vector{Float64}`: Per-SNP normalized variance scalars derived from annotations
- `G::Matrix{Float64}`: Matrix of functional annotations (P × K)
- `γ::Vector{Int}`: Binary indicators of causal status (1=causal, 0=non-causal)
- `function_choices::Vector{Int}`: Selected functional forms for each annotation
- `phi::Vector{Function}`: Functions mapping each annotation to effect size variance
- `sigma_squared::Vector{Float64}`: Per-SNP variance components before normalization

# Details
This function simulates genotype and phenotype data with a realistic genetic architecture:

1. Genotypes (X) are simulated with a realistic LD structure using random eigendecomposition
2. 10% of variants are designated as causal (γ)
3. Functional annotations (G) influence effect size variance through different functional forms
4. Effect sizes follow a spike-and-slab distribution:
   - Causal variants: Normal(0, scaled_variance)
   - Non-causal variants: Normal(0, tiny_variance)
5. Phenotypes are computed as Y = Xβ + ε, with noise calibrated to achieve target heritability

The simulation includes both continuous and binary annotations, with varying relationships
to effect size variance (linear, quadratic, or null effects).
"""
function simulate_raw(;N = 10_000, P = 1_000, K = 100, h2 = 0.10)

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
    return (; X, β, Y, Σ, s, G, γ, function_choices, phi, sigma_squared)
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
    mse = map(i -> var(Y .- view(coef, i) .* view(X, :, i)), 1:P)
    SE = sqrt.(mse ./ D)
    Z = coef ./ SE
    R = cor(X)
    return (; coef, SE, Z, R, D)
end
