module SimpleMctsTests

using Test
using JET
using ReinforcementLearningBase
using ReinforcementLearningEnvironments
using Random: MersenneTwister
using Statistics: mean

using ...SimpleMcts
using ..Common.TestEnvs

export run_mcts_tests

function random_walk_value(; N, start_pos, rep=1000)
    env = RandomWalk1D(; start_pos, N)
    rng = MersenneTwister(0)
    oracle = RolloutOracle(rng)
    return mean(oracle(env)[2] for _ in 1:rep)
end

function uniform_mcts_policy_tic_tac_toe(; n=100)
    return Policy(;
        num_simulations=n,
        num_considered_actions=9,
        value_scale=0.1,
        max_visit_init=50,
        oracle=uniform_oracle,
    )
end

function profile_rollout()
    oracle = RolloutOracle(MersenneTwister(0))
    env = TicTacToeEnv()
    for _ in 1:100
        oracle(env)
    end
end

function profile_explore()
    policy = uniform_mcts_policy_tic_tac_toe()
    env = tictactoe_winning()
    for _ in 1:100
        explore(policy, env)
    end
end

function run_mcts_tests()
    @testset "mcts oracle" begin
        @testset "RolloutOracle" begin
            @test isapprox(
                [random_walk_value(; N=5, start_pos=i) for i in 2:4],
                [-0.5, 0, 0.5],
                atol=0.1,
            )

            nb_actions = 2
            rollout_oracle = RolloutOracle(MersenneTwister(0))
            @test rollout_oracle(RandomWalk1D(; N=5))[1] == ones(nb_actions) ./ nb_actions
        end

        @testset "uniform_oracle" begin
            @test uniform_oracle(RandomWalk1D(; N=5))[2] == 0

            nb_actions = 2
            @test uniform_oracle(RandomWalk1D(; N=5))[1] == ones(nb_actions) ./ nb_actions
        end
    end
    @testset "mcts policy" begin
        policy = uniform_mcts_policy_tic_tac_toe()
        env = tictactoe_winning()
        tree = explore(policy, env)
        qvalue_list = completed_qvalues(tree)
        best = argmax(qvalue_list)

        best_move = 3
        @test length(qvalue_list) ==
            length(tree.children) ==
            length(legal_action_space(env))
        @test legal_action_space(env)[best] == best_move
    end
    @testset "mcts inferred" begin
        @test_opt target_modules = (SimpleMcts,) profile_rollout()
        @test_opt target_modules = (SimpleMcts,) profile_explore()
    end
end

end
