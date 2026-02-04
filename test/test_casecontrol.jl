using Test
using PRSFNN: standardize_beta, compute_effective_sample_size
using DataFrames

@testset "Case-Control Functionality" begin
    @testset "Effective Sample Size" begin
        # Test balanced design
        N_eff = compute_effective_sample_size(5000, 5000)
        @test N_eff ≈ 10000.0

        # Test unbalanced design
        N_eff = compute_effective_sample_size(2000, 8000)
        @test N_eff ≈ 6400.0

        # Test vector inputs
        N_case = [5000, 2000, 3000]
        N_control = [5000, 8000, 7000]
        N_eff = compute_effective_sample_size(N_case, N_control)
        @test N_eff[1] ≈ 10000.0
        @test N_eff[2] ≈ 6400.0
        @test N_eff[3] ≈ 8400.0
    end

    @testset "Beta Standardization - Quantitative" begin
        # Test quantitative trait standardization (original functionality)
        BETA = [0.01, 0.02, 0.015]
        SE = [0.005, 0.006, 0.005]
        N = [10000, 10000, 10000]
        MAF = [0.3, 0.2, 0.4]

        BETA_std = standardize_beta(BETA, SE, N, MAF; trait_type = "quantitative")

        # Check that standardization produces reasonable values
        @test length(BETA_std) == length(BETA)
        @test all(isfinite.(BETA_std))
        @test !any(isnan.(BETA_std))
    end

    @testset "Beta Standardization - Case-Control" begin
        # Test case-control trait standardization
        # log-OR from a case-control study
        BETA = [0.1, 0.15, 0.12]  # log-ORs
        SE = [0.05, 0.06, 0.055]
        N = [8000, 7500, 8200]  # Effective sample sizes
        MAF = [0.3, 0.2, 0.4]
        prevalence = 0.1  # 10% disease prevalence

        BETA_std = standardize_beta(BETA, SE, N, MAF; trait_type = "case-control", prevalence = prevalence)

        # Check that liability scale conversion produces reasonable values
        @test length(BETA_std) == length(BETA)
        @test all(isfinite.(BETA_std))
        @test !any(isnan.(BETA_std))

        # Liability scale betas should be smaller than log-ORs due to conversion
        # (generally true for common diseases with low prevalence)
        @test all(abs.(BETA_std) .< abs.(BETA))
    end

    @testset "Beta Standardization - Prevalence Sensitivity" begin
        # Test that different prevalences give different results
        BETA = [0.1]
        SE = [0.05]
        N = [8000]
        MAF = [0.3]

        BETA_std_01 = standardize_beta(BETA, SE, N, MAF; trait_type = "case-control", prevalence = 0.1)
        BETA_std_05 = standardize_beta(BETA, SE, N, MAF; trait_type = "case-control", prevalence = 0.5)

        # Different prevalences should yield different standardized betas
        @test BETA_std_01[1] != BETA_std_05[1]

        # For rare diseases (low prevalence), conversion factor should be larger
        @test abs(BETA_std_01[1] / BETA[1]) < abs(BETA_std_05[1] / BETA[1])
    end
end
