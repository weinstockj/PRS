using Test
using PRSFNN: joint_log_prob, rss, elbo, simulate_raw, estimate_sufficient_statistics, train_until_convergence, fit_heritability_nn
using Distributions: Normal
using Statistics: cor
using TimerOutputs
using PDMats
using LinearAlgebra
using Flux

@testset "tests" begin
    
    #@test rss(
    #    [0.0011, .0052, 0.0013],
    #    [-0.019, 0.013, -.0199],
    #    [.0098, .0098, .0102],
    #    [1.0 .03 .017; .031 1.0 -0.03; .017 -0.02 1.0]
    #) == 6.551876500157087

    @test rss(
	      [0.0011, .0052, 0.0013],
	      [-0.019, 0.013, -.0199],
	      PDMat(Hermitian([1.0 .03 .017; .031 1.0 -0.03; .017 -0.02 1.0]) + 1e-8 * I),
	      [1.0 0.03 0.0163333; 0.031 1.0 -0.0288235; 0.0176939 -0.0208163 1.0],
	      TimerOutput()
    ) == -2.756206262666406

    # @test abs(joint_log_prob(
    #    [0.0011, .0052, 0.0013],
    #    [-0.019, 0.013, -.0199],
    #    [.0098, .0098, .0102],
    #    [1.0 .03 .017; .031 1.0 -0.03; .017 -0.02 1.0],
    #    [0.01, 0.01, 0.01],
    #    [0.10, 0.10, 0.10]
    #   ) - 15.95427) < .0001

    @test abs(joint_log_prob(
        [0.0011, .0052, 0.0013],
        [-0.019, 0.013, -.0199],
        PDMat(Hermitian([1.0 .03 .017; .031 1.0 -0.03; .017 -0.02 1.0]) + 1e-8 * I),
        [1.0 0.03 0.0163333; 0.031 1.0 -0.0288235; 0.0176939 -0.0208163 1.0],
        [0.01, 0.01, 0.01],
        [0.10, 0.10, 0.10],
        TimerOutput()
    ) - 6.64619 ) < .0001

    # @test abs(elbo(
    #     rand(Normal(0, 1), 3),
    #     [0.01, -0.003, 0.0018],
    #     [-9.234, -9.24, -9.24],
    #     [0.023, -0.0009, -.0018],
    #     [.0094, .00988, .0102],
    #     [1.0 .03 .017; .031 1.0 -0.03; .017 -0.02 1.0],
    #     0.01,
    #     0.10
    #    ) - -5.10) < 0.1
    
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

        #trained_model = PRSFNN.fit_heritability_nn(
        trained_model = fit_heritability_nn(
                model, 
                q_var, 
                q_α, 
                G;
                patience = 100
        )

        yhat = transpose(trained_model(transpose(G)))
        yhat[:, 1] .= exp.(yhat[:, 1])
        yhat[:, 2] .= 1.0 ./ (1.0 .+ exp.(-yhat[:, 2]))

        @test round(cor(yhat[:, 1], q_var), digits=1) >= .8
        @test round(cor(yhat[:, 2], q_α), digits=1) >= .8
    end

    test_nn()

    function test_complete_run()
        K = 100
        model = Chain(
                Dense(K => 5, relu; init = Flux.glorot_normal(gain = 0.0005)),
                Dense(5 => 2)
        )
        raw = simulate_raw(;N = 10_000, P = 1000, K = 100)
        ss = estimate_sufficient_statistics(raw[1], raw[3])
        out = train_until_convergence(ss[1], 
				       ss[2], 
				       convert(AbstractArray{Float64}, ss[4]), 
				       ss[5], 
				       raw[6], 
				       model = model, 
				       N = fill(10_000, length(ss[1])))
        @test round(cor(out[1] .* out[2], raw[2]), digits=1) >= 0.6
    end

    test_complete_run()

        
end
