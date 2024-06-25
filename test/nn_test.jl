function test_long_nn()
    P = 800
    K = 30
    H = 3

    # model = Chain(
    #         Dense(K => 3, relu; init = Flux.glorot_normal(gain = 0.0001)),
    #         Dense(3 => 2)
    # )
    layer_1 = Dense(K => H, softplus; init = Flux.glorot_normal(gain = 0.001))
    layer_output = Dense(H => 2)
    layer_output.bias .= [StatsFuns.log(0.0001), StatsFuns.logit(0.005)]
    model = Chain(layer_1, layer_output)
    optim_type = AdamW(0.02)
    opt = Flux.setup(optim_type, model)

    G = rand(Normal(0, 1), P, K)
    β = rand(Normal(0, 0.3), K)

    q_var = 0.05 .* exp.(G * β)
    q_α = Flux.σ.(-2.0 .+ 3.0 .* q_var)

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
            mse_improvement_threshold = 0.001,
            learning_rate_decay = 0.95
    )

    # decay of 0.90 has minimum cor of 0.58
    # decay of 0.95 has minimum cor of 0.67
    # decay of 0.80 has minimum cor of 0.59

    yhat = transpose(trained_model(transpose(G)))
    yhat[:, 1] .= exp.(yhat[:, 1])
    yhat[:, 2] .= Flux.σ.(yhat[:, 2])

    PRSFNN.describe_vector(yhat[:, 1])
    PRSFNN.describe_vector(yhat[:, 2])

    var_cor = cor(yhat[:, 1], q_var)
    α_cor  = cor(yhat[:, 2], q_α)
    @test var_cor > 0.75
    @test α_cor > 0.75

    # return α_cor, var_cor
end
