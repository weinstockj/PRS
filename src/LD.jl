"""
    compute_LD(LD_reference::String)

    Compute the LD matrix from a reference genotype file

"""
function compute_LD(LD_reference::String = "test_data/test_data/chr1_16103_1170341/filtered.bed")
    genotypes = SnpArray(LD_reference)
    genotypes_float = convert(Matrix{Float64}, genotypes, impute=true)
    R = cor(genotypes_float)
    D = map(x -> sum(x .^ 2), eachcol(genotypes_float))
    return R, D
end

function poet_cov(X::AbstractArray; K = 10, τ = 0.03, N = 500)
    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5563862/
    Σ = Symmetric(X)
    eig = eigen(Σ, 1, K)
    P = size(X, 2)
    evals = eig.values
    evecs = eig.vectors
    c = (tr(Σ) - sum(evals)) / (P - K - P * K / N)
    Σk = evecs * Diagonal(evals .- c * P / N) * evecs'
    Σu = Σ - Σk
    @inbounds @simd for i in eachindex(Σu)
        if abs(Σu[i]) < τ
            Σu[i] = 0.0
        end
    end
    return Σk + Σu
end

