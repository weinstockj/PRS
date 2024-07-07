using Test
using PRSFNN: joint_log_prob, log_prior, rss, elbo, simulate_raw, estimate_sufficient_statistics, train_until_convergence, fit_heritability_nn, infer_σ2, construct_XtX, construct_D, construct_Xty
using Distributions: Normal
using Statistics: cor
using TimerOutputs
using PDMats
using LinearAlgebra
using Flux
using StatsFuns


function test_rss_elbo()

    β = [0.0011, .0052, 0.0013]
    coef = [-0.019, 0.013, -.0199] 
    SE = [.0098, .0098, .0102]
    R = [1.0 .03 .017; .031 1.0 -0.03; .017 -0.02 1.0]

    Σ = PDMat(Hermitian(SE .* R .* SE'))
    SRSinv = SE .* R .* (1 ./ SE)

    σ2_β = [0.01, 0.01, 0.01]
    p_causal = [0.10, 0.10, 0.10]
    σ2 = 1.0

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
        σ2,
        TimerOutput()
    ) - 3.792569404258637) < .0001

    @test abs(log_prior(
                β,
                σ2_β,
                p_causal,
                σ2,
                TimerOutput()
            ) - -2.758314098116269) < .0001
end


function test_complete_run()
    K = 100
    H = 5
    layer_1 = Dense(K => H, Flux.softplus; init = Flux.glorot_normal(gain = 0.005))
    layer_output = Dense(H => 2)
    layer_output.bias .= [StatsFuns.log(0.0001), StatsFuns.logit(0.005)]
    model = Chain(layer_1, layer_output)
    optim_type = AdamW(0.02)
    opt = Flux.setup(optim_type, model)
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
               opt = opt,
               N = N_vec
            )
    @test cor(out[1] .* out[2], raw[2]) >= 0.70
end

function test_nn()

    P = 300
    K = 50
    H = 5

    layer_1 = Dense(K => H, Flux.softplus; init = Flux.glorot_normal(gain = 0.001))
    layer_output = Dense(H => 2)
    layer_output.bias .= [StatsFuns.log(0.0001), StatsFuns.logit(0.005)]
    model = Chain(layer_1, layer_output)
    optim_type = AdamW(0.02)
    opt = Flux.setup(optim_type, model)

    G = rand(Normal(0, 1), P, K)

    q_var = 0.1 .* exp.((G * rand(Normal(0, 0.3), K))) 
    q_α = Flux.σ(-2.0 .+ q_var)

    trained_model = fit_heritability_nn(
            model, 
            opt,
            Float32.(q_var), 
            Float32.(q_α), 
            Float32.(G);
            patience = 150
    )

    yhat = transpose(trained_model(Float32.(transpose(G))))
    yhat[:, 1] .= exp.(yhat[:, 1])
    yhat[:, 2] .= Flux.σ(yhat[:, 2])

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
        R = ss[4]
        XtX = construct_XtX(R, X_sd, N)
        D = construct_D(XtX)
        Xty = construct_Xty(ss[1], D)
        σ2, R2, yty = infer_σ2(ss[1], ss[2], XtX, Xty, N, P; estimate = true)
        @test abs(R2 - h2) < 0.05
end

@testset "tests" begin
    test_rss_elbo()
    test_nn()
    test_complete_run()
    test_infer_σ2()
end
