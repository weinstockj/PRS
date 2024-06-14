function test_long_nn()
    P = 500
    K = 20

    # model = Chain(
    #         Dense(K => 3, relu; init = Flux.glorot_normal(gain = 0.0001)),
    #         Dense(3 => 2)
    # )
    model = Chain(Dense(K => 3, relu), Dense(3 => 2))
    optim_type = AdamW(0.02)
    opt = Flux.setup(optim_type, model)

    G = rand(Normal(0, 1), P, K)
    β = rand(Normal(0, 0.3), K)

    q_var = 0.05 .* exp.(G * β)
    q_α = 1.0 ./ (1.0 .+ exp.(-1.0 .* (-2.0 .+ q_var)))

    PRSFNN.describe_vector(q_var)
    PRSFNN.describe_vector(q_α)

    trained_model = PRSFNN.fit_heritability_nn(
            model, 
            opt,
            q_var, 
            q_α, 
            G;
            max_epochs = 4000,
            patience = 400,
            mse_improvement_threshold = 0.01
    )

    yhat = transpose(trained_model(transpose(G)))
    yhat[:, 1] .= exp.(yhat[:, 1])
    yhat[:, 2] .= 1.0 ./ (1.0 .+ exp.(-yhat[:, 2]))

    PRSFNN.describe_vector(yhat[:, 1])
    PRSFNN.describe_vector(yhat[:, 2])

    @test cor(yhat[:, 1], q_var) > .7
    @test cor(yhat[:, 2], q_α) > .7
end
