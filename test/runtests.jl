using Test
using PRSFNN: joint_log_prob, log_prior, rss, elbo, simulate_raw, estimate_sufficient_statistics, train_until_convergence, fit_heritability_nn, infer_σ2
using Distributions: Normal
using Statistics: cor
using TimerOutputs
using PDMats
using LinearAlgebra
using Flux


function test_rss_elbo()

    β = [0.0011, .0052, 0.0013]
    coef = [-0.019, 0.013, -.0199] 
    SE = [.0098, .0098, .0102]
    R = [1.0 .03 .017; .031 1.0 -0.03; .017 -0.02 1.0]

    Σ = PDMat(Hermitian(SE .* R .* SE'))
    SRSinv = SE .* R .* (1 ./ SE)

    σ2_β = [0.01, 0.01, 0.01]
    p_causal = [0.10, 0.10, 0.10]

    @test abs(
            rss(
              β,
              coef,
              Σ,
              SRSinv,
              TimerOutput()
            ) -6.55088350237490
        ) < 1e-8

    @test abs(joint_log_prob(
        β,
        coef,
        Σ,
        SRSinv,
        σ2_β,
        p_causal,
        TimerOutput()
    ) - 3.792569404258637) < .0001

    @test abs(log_prior(
                β,
                σ2_β,
                p_causal,
                TimerOutput()
            ) - -2.758314098116269) < .0001
end


function test_complete_run()
    K = 100
    model = Chain(
            Dense(K => 5, relu; init = Flux.glorot_normal(gain = 0.0005)),
            Dense(5 => 2)
    )
    raw = simulate_raw(;N = 10_000, P = 1000, K = 100)
    ss = estimate_sufficient_statistics(raw[1], raw[3])
    N_vec = fill(10_000, length(ss[1]))
    out = train_until_convergence(
               ss[1], 
               ss[2], 
               convert(AbstractArray{Float64}, ss[4]), 
               ss[5], 
               raw[6], 
               model = model, 
               N = N_vec
            )
    @test cor(out[1] .* out[2], raw[2]) >= 0.6
end

function test_nn()

    P = 300
    K = 50

    model = Chain(
            Dense(K => 5, relu; init = Flux.glorot_normal(gain = 0.0005)),
            Dense(5 => 2)
    )

    G = rand(Normal(0, 1), P, K)

    q_var = 0.1 .* exp.((G * rand(Normal(0, 0.3), K))) 
    q_α = 1.0 ./ (1.0 .+ exp.(-1.0 .* (-2.0 .+ q_var)))

    trained_model = fit_heritability_nn(
            model, 
            Float32.(q_var), 
            Float32.(q_α), 
            Float32.(G);
            patience = 150
    )

    yhat = transpose(trained_model(Float32.(transpose(G))))
    yhat[:, 1] .= exp.(yhat[:, 1])
    yhat[:, 2] .= 1.0 ./ (1.0 .+ exp.(-yhat[:, 2]))

    @test cor(yhat[:, 1], q_var) >= .70
    @test cor(yhat[:, 2], q_α) >= .70
end

function test_infer_σ2()

        N = 10_000 
        P = 200
        K = 100
        h2 = 0.30
        raw = simulate_raw(;N = N, P = P, K = K, h2 = h2)
        ss = estimate_sufficient_statistics(raw[1], raw[3])
        X_sd = sqrt.(ss[5] ./ N)
        σ2, R2, yty = infer_σ2(ss[1], ss[2], ss[4], ss[5], X_sd, N, P)
        @test abs(R2 - h2) < 0.05
end

@testset "tests" begin
    test_rss_elbo()
    test_nn()
    test_complete_run()
end
