function test_long_nn()
    P = 100
    K = 20

    model = Chain(
            Dense(K => 5, relu; init = Flux.glorot_normal(gain = 0.0005)),
            Dense(5 => 2)
    )

    G = rand(Normal(0, 1), P, K)
    β = rand(Normal(0, 0.3), K)

    q_var = 0.1 .* exp.(G * β)
    q_α = 1.0 ./ (1.0 .+ exp.(-1.0 .* (-2.0 .+ q_var)))

    trained_model = PRSFNN.fit_heritability_nn(
            model, 
            q_var, 
            q_α, 
            G;
            patience = 100
    )

    yhat = transpose(trained_model(transpose(G)))
    yhat[:, 1] .= exp.(yhat[:, 1])
    yhat[:, 2] .= 1.0 ./ (1.0 .+ exp.(-yhat[:, 2]))

    @test cor(yhat[:, 1], q_var) > .7
    @test cor(yhat[:, 2], q_α) > .7
end
