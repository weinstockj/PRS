"""
    compute_LD(LD_reference::String)

Compute linkage disequilibrium (LD) correlation matrix from a reference genotype file.

# Arguments
- `LD_reference::String`: Path to the BED format genotype file

# Returns
- `R::Matrix{Float64}`: LD correlation matrix for polymorphic variants
- `sds::Vector{Float64}`: Standard deviations of genotypes for each variant
- `allele_freq::Vector{Float64}`: Allele frequencies (mean/2) for each variant
- `good_variants::Vector{Int}`: Indices of polymorphic variants that passed QC filters

# Details
This function reads genotype data from a BED file, converts it to floating point representation,
filters out monomorphic variants and those with very low variance, and computes the correlation
matrix between the remaining variants. Variants are filtered based on mean frequency and standard
deviation thresholds.
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

"""
    construct_XtX(R::AbstractArray, X_sd::Vector{Float64}, N::Real)

Construct X'X matrix for genetic association analysis.

# Arguments
- `R::AbstractArray`: Correlation matrix of genotypes
- `X_sd::Vector{Float64}`: Standard deviations of genotype values
- `N::Real`: Sample size (typically the mean or median sample size from GWAS)

# Returns
- `XtX::Symmetric{Float64}`: The X'X matrix scaled by sample size

# Details
This function converts a correlation matrix to a covariance matrix using
standard deviations, then scales by the sample size. This matrix represents
X'X in the standard linear regression equation (X'X)β = X'y.
"""
function construct_XtX(R::AbstractArray, X_sd::Vector{Float64}, N::Real)
    return Symmetric(X_sd .* R .* X_sd') .* N
    # return Symmetric(R) .* N
end

"""
    construct_Xty(coef::Vector{Float64}, D)

Construct X'y vector for genetic association analysis.

# Arguments
- `coef::Vector{Float64}`: Vector of effect sizes (typically GWAS beta coefficients)
- `D`: Diagonal elements of X'X matrix

# Returns
- `Xty::Vector{Float64}`: The X'y vector, computed as coefficient * diagonal

# Details
This function creates the X'y vector by multiplying each coefficient by the
corresponding diagonal element of X'X. This is used in the standard linear
regression equation (X'X)β = X'y.
"""
function construct_Xty(coef::Vector{Float64}, D)
    return coef .* D
end

"""
    construct_D(XtX::AbstractArray)

Extract the diagonal elements from X'X matrix.

# Arguments
- `XtX::AbstractArray`: X'X matrix from which to extract diagonal elements

# Returns
- `D::SubArray`: A view of the diagonal elements of XtX

# Details
This function returns a view (not a copy) of the diagonal elements of the
input matrix. Using a view is more memory efficient than creating a new array.
"""
function construct_D(XtX::AbstractArray)
    return @view(XtX[diagind(XtX)])
end

"""
    poet_cov(X::AbstractArray; K = 100, τ = 0.01, N = 1000)

Apply POET (Principal Orthogonal complEment Thresholding) method to estimate a sparse covariance matrix.

# Arguments
- `X::AbstractArray`: Input covariance or correlation matrix
- `K::Int=100`: Number of principal components to retain
- `τ::Float64=0.01`: Thresholding parameter for sparsity
- `N::Int=1000`: Sample size used for bias correction

# Returns
- `Sigma::Matrix{Float64}`: The POET-estimated covariance matrix

# Details
Implements the POET method for high-dimensional covariance matrix estimation.
The method decomposes the matrix into a low-rank component (representing systematic factors)
and a sparse component (representing idiosyncratic noise). The sparse component
is obtained by thresholding.

Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5563862/
"""
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

