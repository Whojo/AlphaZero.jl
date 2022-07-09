module BitwiseRandomWalk1DTests

using Test
using ..BatchedEnvsTests
using ..Common.BitwiseRandomWalk1D
using ReinforcementLearningEnvironments

export run_BitwiseRandomWalk1D_tests

function run_BitwiseRandomWalk1D_tests()
    @testset "bitwise RandomWalk1D" begin
        test_equivalent(BitwiseRandomWalk1DEnv, RandomWalk1D)
        test_batch_simulate(BitwiseRandomWalk1DEnv)
    end
    return nothing
end

end
