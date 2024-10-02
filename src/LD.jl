"""
    compute_LD(LD_reference::String)

    Compute the LD matrix from a reference genotype file

"""
function compute_LD(LD_reference::String = "test_data/test_data/chr1_16103_1170341/filtered.bed")
    genotypes = SnpArray(LD_reference)
    # rows are samples, columns are SNPs
    genotypes_float = convert(Matrix{Float64}, genotypes, impute=true)
    mean_frequencies = vec(mean(genotypes_float; dims = 1))
    sds              = vec(std(genotypes_float; dims = 1))
    good_variants = findall((mean_frequencies .> 0.0) .& (mean_frequencies .< 2.0) .& (sds .> .01)) # remove monomorphic variants and those with no variance
    @info "$(ltime()) Number of polymorphic variants out of all variants: $(length(good_variants)) / $(size(genotypes_float, 2))"
    good_genotypes = view(genotypes_float, :, good_variants)
    R = cor(good_genotypes)
    return R, sds, mean_frequencies ./ 2.0, good_variants
end

function construct_XtX(R::AbstractArray, X_sd::Vector{Float64}, N::Real)
    return Symmetric(X_sd .* R .* X_sd') .* N
    # return Symmetric(R) .* N
end

function construct_Xty(coef::Vector{Float64}, D)
    return coef .* D
end

function construct_D(XtX::AbstractArray)
    return @view(XtX[diagind(XtX)])
end

function poet_cov(X::AbstractArray; K = 100, τ = 0.01, N = 1000)
    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5563862/
    Σ = Symmetric(X)
    P = size(X, 2)
    eig = eigen(Σ, (P - K):P)
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
    # return Σk + Σu, Σk, Σu
end

