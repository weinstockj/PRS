using Test
using LinearAlgebra
using PRSFNN: inner_loop_cavi!, inner_loop_cavi_fast!, infer_σ2, simulate_raw, estimate_sufficient_statistics, construct_XtX, construct_D, construct_Xty

function test_inner_loop_cavi()

    N = 10_000 
    P = 500
    K = 50
    h2 = 0.20
    raw = simulate_raw(;N = N, P = P, K = K, h2 = h2)
    ss = estimate_sufficient_statistics(raw.X, raw.Y)
    X_sd = sqrt.(ss.D ./ N)

    XtX = construct_XtX(ss.R, X_sd, N)
    Xty = construct_Xty(ss.coef, ss.D) 
       # Create small, deterministic test data
    q_μ1 = zeros(P)
    q_μ2 = zeros(P)
    q_spike_μ1 = zeros(P)
    q_spike_μ2 = zeros(P)
    q_α1 = fill(0.1, P)
    q_α2 = fill(0.1, P)
    q_var1 = fill(0.05, P)
    q_var2 = fill(0.05, P)
    q_spike_var1 = fill(1e-8, P)
    q_spike_var2 = fill(1e-8, P)

    σ2, R2, yty = infer_σ2(ss.coef, ss.SE, XtX, Xty, N, P; estimate = true)
    # Run the function
    inner_loop_cavi!(q_μ1, q_spike_μ1, q_α1, q_var1, q_spike_var1, XtX, Xty, σ2; P = P)
    inner_loop_cavi_fast!(q_μ2, q_spike_μ2, q_α2, q_var2, q_spike_var2, XtX, Xty, σ2; P = P)

    # Check that the arrays have changed from their initial values
    @test q_μ1 ≈ q_μ2
    @test q_spike_μ1 ≈ q_spike_μ2
    @test q_α1 ≈ q_α2
    @test q_var1 ≈ q_var2

    @show q_μ1[1:5], q_μ2[1:5]
end

