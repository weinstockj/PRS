using Test
using PRSFNN: joint_log_prob, rss, elbo, simulate_raw, estimate_sufficient_statistics, train_until_convergence
using Distributions: Normal
using Statistics: cor

@testset "tests" begin
    
    @test rss(
        [0.0011, .0052, 0.0013],
        [-0.019, 0.013, -.0199],
        [.0098, .0098, .0102],
        [1.0 .03 .017; .031 1.0 -0.03; .017 -0.02 1.0]
    ) == 6.551876500157087

    @test abs(joint_log_prob(
        [0.0011, .0052, 0.0013],
        [-0.019, 0.013, -.0199],
        [.0098, .0098, .0102],
        [1.0 .03 .017; .031 1.0 -0.03; .017 -0.02 1.0],
        0.01,
        0.10
       ) - 15.95427) < .0001

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
    function test_complete_run()
        raw = simulate_raw()
        ss = estimate_sufficient_statistics(raw[1], raw[3])
        out = train_until_convergence(ss[1], ss[2], ss[4], ss[5], raw[6])
        @test cor(out[1] .* out[2], raw[2]) > 0.7
    end

    test_complete_run()
end
