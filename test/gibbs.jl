using Statistics: std
using TimerOutputs: TimerOutput

function test_gibbs(max_iter = 300)
    
    raw = PRSFNN.simulate_raw(;N = 10_000, P = 1000, K = 100)
    # return X, β, Y, Σ, s, G, γ, function_choices, phi, sigma_squared
    ss = PRSFNN.estimate_sufficient_statistics(raw[1], raw[3])
    # return coef, SE, Z, cor(X), D
    N_vec = fill(10_000, length(ss[1]))

    X_sd = vec(std(raw[1], dims = 1))

    XtX = PRSFNN.construct_XtX(ss[4], X_sd, first(N_vec))
    D = PRSFNN.construct_D(XtX)
    Xty = PRSFNN.construct_Xty(ss[1], D)
    to = TimerOutput()
# function train_gibbs(p_causal, σ2_β, coef, SE, R, XtX, Xty, to; P = 1_000, max_iter = 5, N = 10_000, yty = 1.0, spike_σ2 = 1e-8)

#
# function infer_σ2(coef::Vector, SE::Vector, XtX::AbstractArray, Xty::Vector, N::Real, P::Int64; estimate = false, λ = 100)
    
    σ2, R2, yty = PRSFNN.infer_σ2(ss[1], ss[2], XtX, Xty, first(N_vec), length(ss[1]); estimate = true)

    gibbs_est = PRSFNN.train_gibbs(
        0.10 .* ones(length(ss[1])),
        0.01 .* ones(length(ss[1])),
        ss[1],
        ss[2],
        ss[4],
        XtX,
        Xty,
        to;
        max_iter = max_iter,
        yty = yty
    )

    cavi_est = PRSFNN.train_cavi(
        0.10 .* ones(length(ss[1])),
        0.01 .* ones(length(ss[1])),
        ss[1],
        ss[2],
        ss[4],
        XtX,
        Xty,
        to;
        max_iter = 5,
        yty = yty
    )

    show(to)
    println()

    return (
        β = raw[2], 
        gibbs_β = gibbs_est[1], 
        gibbs_α = gibbs_est[2], 
        cavi_β  = cavi_est[1],
        cavi_α  = cavi_est[2]
    )

end
