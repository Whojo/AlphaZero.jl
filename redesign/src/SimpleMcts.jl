"""
A straightforward non-batched implementation of Gumbel MCTS.

# Design choices
- We represent all available actions explicitly for each node.
- We reset the tree everytime (for now).
- All values are from the current player perspective.
- All computations are done using `Float64` but `Float32` is accepted from oracles.
- It is the responsability of the oracle to only provides valid actions.
"""
module SimpleMcts

using ReinforcementLearningBase
using Distributions: sample, Gumbel
using Random: AbstractRNG
using Flux: softmax
using Base: @kwdef

export Policy, gumbel_explore, explore, completed_qvalues
export RolloutOracle, uniform_oracle

"""
An MCTS tree.

# Attributes
- `oracle_value::Float32`: value estimated by the oracle (e.g. `uniform_oracle`) when a
     `Tree` is created.
- `children::Vector{Union{Nothing,Tree}}`: children of the current `Tree`. The indexes
    of each child correspond to the one given by `legal_action_space(env)`, with `env` the
    implicit environment (as defined in ReinforcementLearningEnvironment.jl) associated
    with this `Tree` during exploration (i.e. `explore` function).
- `prior::Vector{Float32}`: prior probabilities estimated by the oracle
     (e.g. `uniform_oracle`) when a `Tree` is created.
- `num_visits::Vector{Int32}`: number of visits of the current `Tree` during the Mcts
    exploration (i.e. `explore` function).
- `total_rewards::Vector{Float64}`: sum of the accumulated rewards of the current 
    `Tree` during the Mcts exploration (i.e. `explore` function).

# Notes
This MCTS tree is represented by tree of structures in memory.
We store Q-values for each nodes instead of storing values
so as to make it easier to handle terminal states.
"""
mutable struct Tree
    oracle_value::Float32
    children::Vector{Union{Nothing,Tree}}
    prior::Vector{Float32}
    num_visits::Vector{Int32}
    total_rewards::Vector{Float64}
end

"""
An MCTS Policy that leverages an external oracle.

# Attributes
- `oracle::Oracle`: function that, given an environment (as defined in
    ReinforcementLearningEnvironment.jl), computes the prior probabilities associated with
    its actions (children of a Mcts `Tree`) and the value associated with the provided
    environment.
- `num_simulations::Int = 64`: number of simulations to run on the given Mcts `Tree`.
- `num_considered_actions::Int = 8`: number of actions considered by gumbel during
    exploration. Only the `num_conidered_actions` actions with the highest `value_prior`
    probabilities and gumbel noise will be used. It should be a power of 2.
- `value_scale::Float64 = 0.1f0`: multiplying coefficient to weight the qvalues against the
    prior probabilities during exploration. Prior probabilities have, by default, a
    decreasing weight when the number of visits increase.
- `max_visit_init::Int = 50`: artificial increase of the number of visits to weight qvalue
    against prior probabilities on low visit count.

# Notes
The attributes `num_conidered_actions`, `value_scale` and `max_visit_init` are specific to
the gumbel implementation.
"""
@kwdef struct Policy{Oracle}
    oracle::Oracle
    num_simulations::Int = 64
    num_considered_actions::Int = 8
    value_scale::Float64 = 0.1f0
    max_visit_init::Int = 50
end

"""
    function num_children(node::Tree)
    
Number of `children` associated to a given `node`.
"""
num_children(node::Tree) = length(node.children)

"""
    function max_child_visits(node::Tree)

Maximum `num_visits` among the `children` of a given `node`.
"""
max_child_visits(node::Tree) = maximum(node.num_visits; init=0)

"""
    function visited_children_indices(node::Tree)
        
Return, for a given `node`, the indices of the `children` that have been visited at least
once during exploration.
"""
function visited_children_indices(node::Tree)
    return (i for i in eachindex(node.children) if node.num_visits[i] > 0)
end

"""
    function children_indices(node::Tree)
    
Return the indices of each `children` of a given `node`.
"""
children_indices(node::Tree) = eachindex(node.children)

"""
    function qvalue(node::Tree, i)

Compute the qvalue for an action `i` from a given `node`.
"""
function qvalue(node::Tree, i)
    n = node.num_visits[i]
    @assert n > 0
    return node.total_rewards[i] / n
end

"""
    function qcoeff(mcts::Policy, node::Tree)

Compute the qcoeff for a given Gumbel `mcts` policy and a given `node`.
    
# Notes
The paper introduces a sigma function, which we implement by sigma(q) = qcoeff * q
"""
function qcoeff(mcts::Policy, node::Tree)
    return mcts.value_scale * (mcts.max_visit_init + max_child_visits(node))
end

"""
    function root_value_estimate(node::Tree)

Compute the root value for a given `node`.

Its value depends on the `num_visits` and `qvalue` of all of its children combined with its
`oracle_value` and `prior` estimate. See the code for more details.
"""
function root_value_estimate(node::Tree)
    total_visits = sum(node.num_visits)
    root_value = node.oracle_value
    visited = collect(visited_children_indices(node))
    if !isempty(visited)
        children_value = sum(node.prior[i] * qvalue(node, i) for i in visited)
        children_value /= sum(node.prior[i] for i in visited)
    else
        children_value = 0.0
    end
    return (root_value + total_visits * children_value) / (1 + total_visits)
end

"""
    function completed_qvalues(node::Tree)

Return a list of estimated qvalue of all children for a given `node`.

More precisely, if its child have been visited at least one time, it computes its real
`qvalue`, otherwise, it uses the `root_value_estimate` of `node` instead.

# Example
```jldoctest
julia> using RLZero, ReinforcementLearningBase
julia> using .Tests

julia> policy = SimpleMctsTests.uniform_mcts_policy()
[...]

julia> env = TestEnvs.tictactoe_winning()
[...]
x . .
. x .
. o o

julia> tree = explore(policy, env)
[...]

julia> qvalues = SimpleMcts.completed_qvalues(tree) # How to use `completed_qvalues`
5-element Vector{Float64}:
 -1.0
  0.9595959595959596
 -0.020002000200020006
 -0.020002000200020006
 -0.020002000200020006

julia> best_move = argmax(qvalues)
2

julia> env(legal_action_space(env)[best_move]); env # Play the best move
[...]
x . .
. x .
x o o
```
"""
function completed_qvalues(node::Tree)
    root_value = root_value_estimate(node)
    return map(children_indices(node)) do i
        node.num_visits[i] > 0 ? qvalue(node, i) : root_value
    end
end

"""
    function create_node(env::AbstractEnv, oracle)
    
Create a new `Tree` node for a given environment `env` and an `oracle`.
"""
function create_node(env::AbstractEnv, oracle)
    prior, oracle_value = oracle(env)
    num_actions = length(legal_action_space(env))
    @assert num_actions > 0
    children = convert(Vector{Union{Nothing,Tree}}, fill(nothing, num_actions))
    num_visits = fill(Int32(0), num_actions)
    total_rewards = fill(Float64(0), num_actions)
    return Tree(prior, oracle_value, children, num_visits, total_rewards)
end

"""
    function gumbel_explore(mcts::Policy, env::AbstractEnv, rng::AbstractRNG)

Run MCTS search with Gumbel exploration noise on the current environment `env`
with a given `mcts` policy and return an MCTS tree.

The `rng` parameter is here used for replicability.
 
# Example
```jldoctest
julia> using RLZero, ReinforcementLearningBase
julia> using Random: MersenneTwister
julia> using .Tests

julia> policy = SimpleMctsTests.uniform_mcts_policy()
[...]

julia> env = TestEnvs.tictactoe_winning()
[...]
x . .
. x .
. o o

julia> tree = gumbel_explore(policy, env, MersenneTwister(0)) # How to use `gumbel_explore`
[...]

julia> best_move = argmax(SimpleMcts.completed_qvalues(tree))
2

julia> env(legal_action_space(env)[best_move]); env # Play the best move
[...]
x . .
. x .
x o o
```

See also [`explore`](@ref).
"""
function gumbel_explore(mcts::Policy, env::AbstractEnv, rng::AbstractRNG)
    # Creating an empty tree, sampling the Gumbel variables
    # and selecting m actions with top scores.
    node = create_node(env, mcts.oracle)
    gscores = [rand(rng, Gumbel()) for _ in children_indices(node)]
    base_scores = gscores + log.(node.prior)
    num_considered = min(mcts.num_considered_actions, length(node.children))
    @assert num_considered > 0
    considered::Vector = partialsortperm(base_scores, 1:num_considered; rev=true)
    # Sequential halving
    num_prev_sims = 0  # number of performed simulations
    num_halving_steps = Int(ceil(log2(num_considered)))
    sims_per_step = mcts.num_simulations / num_halving_steps
    while true
        num_visits = Int(max(1, floor(sims_per_step / num_considered)))
        for _ in 1:num_visits
            # If we do not have enough simulations left to
            # visit every considered actions, we must visit
            # the most promising ones with higher priority
            if num_prev_sims + num_considered > mcts.num_simulations
                # For the q-values to exist, we need
                # num_simulations > num_conidered_actions
                qs = [qvalue(node, i) for i in considered]
                scores = base_scores[considered] + qcoeff(mcts, node) * qs
                considered = considered[sortperm(scores; rev=true)]
            end
            # We visit all considered actions once
            for i in considered
                run_simulation_from_child(mcts, node, copy(env), i)
                num_prev_sims += 1
                if num_prev_sims >= mcts.num_simulations
                    return node
                end
            end
        end
        # Halving step
        num_considered = max(2, num_considered ÷ 2)
        qs = [qvalue(node, i) for i in considered]
        scores = base_scores[considered] + qcoeff(mcts, node) * qs
        considered = considered[partialsortperm(scores, 1:num_considered; rev=true)]
    end
end

"""
    function explore(mcts::Policy, env::AbstractEnv)

Run MCTS search on the current environment `env` for a given `mcts` policy and return an
MCTS tree.

# Example
```jldoctest
julia> using RLZero, ReinforcementLearningBase
julia> using Random: MersenneTwister
julia> using .Tests

julia> policy = SimpleMctsTests.uniform_mcts_policy()
[...]

julia> env = TestEnvs.tictactoe_winning()
[...]
x . .
. x .
. o o

julia> tree = explore(policy, env, MersenneTwister(0)) # How to use `explore`
[...]

julia> best_move = argmax(SimpleMcts.completed_qvalues(tree))
2

julia> env(legal_action_space(env)[best_move]); env # Play the best move
[...]
x . .
. x .
x o o
```

See also [`gumbel_explore`](@ref).
"""
function explore(mcts::Policy, env::AbstractEnv)
    node = create_node(env, mcts.oracle)
    for _ in 1:(mcts.num_simulations)
        run_simulation(mcts, node, copy(env))
    end
    return node
end

"""
    function run_simulation_from_child(mcts::Policy, node::Tree, env::AbstractEnv, i)
        
Run MCTS search for a given `mcts` policy on the environment `env` with an action `i` played
and return the acquired reward.

This function is used internally and run indirecly by `explore` and `gumbel_explore`.
"""
function run_simulation_from_child(mcts::Policy, node::Tree, env::AbstractEnv, i)
    prev_player = current_player(env)
    actions = legal_action_space(env)
    env(actions[i])
    r = reward(env, prev_player)
    switched = prev_player != current_player(env)
    if is_terminated(env)
        next_value = zero(r)
    else
        if isnothing(node.children[i])
            node.children[i] = create_node(env, mcts.oracle)
        end
        child = node.children[i]
        @assert !isnothing(child)
        next_value = run_simulation(mcts, child, env)
    end
    value = r + (switched ? -next_value : next_value)
    node.num_visits[i] += 1
    node.total_rewards[i] += value
    return value
end

"""
    function run_simulation(mcts::Policy, node::Tree, env::AbstractEnv)

Select an action in a given environment `env` and run a simulation on this action according
to a `mcts` policy, and return the acquired reward.
"""
function run_simulation(mcts::Policy, node::Tree, env::AbstractEnv)
    i = select_nonroot_action(mcts, node)
    return run_simulation_from_child(mcts, node, env, i)
end

"""
    function target_policy(mcts::Policy, node::Tree)

Compute the target policy used to choose an action on `node`, depending on a given `mcts`
and return it as a probability distribution vector.
"""
function target_policy(mcts::Policy, node::Tree)
    qs = completed_qvalues(node)
    return softmax(log.(node.prior) + qcoeff(mcts, node) * qs)
end

"""
    function select_nonroot_action(mcts::Policy, node::Tree)
        
Select the best move to play for a `node`, accordingly to a given `policy`.
"""
function select_nonroot_action(mcts::Policy, node::Tree)
    policy = target_policy(mcts, node)
    total_visits = sum(node.num_visits)
    return argmax(
        policy[i] - node.num_visits[i] / (total_visits + 1) for i in 1:length(node.children)
    )
end

"""
# Some standard oracles
"""

"""
    function uniform_oracle(env::AbstractEnv)

Oracle that always returns a value of 0 and a uniform policy.

Useful for testing and setting up a pipeline. Can be used as oracle in the struct `Policy`.

# Example
```jldoctest
julia> using RLZero, ReinforcementLearningEnvironments
julia> env = RandomWalk1D()
[...]

julia> uniform_oracle(env)
([0.5, 0.5], 0.0)
```

See also [`RolloutOracle`](@ref).
"""
function uniform_oracle(env::AbstractEnv)
    n = length(legal_action_space(env))
    P = ones(n) ./ n
    V = 0.0
    return P, V
end

"""
Oracle that performs a single random rollout to estimate state value.

Given a state, the oracle selects random actions until a leaf node is reached.
The resulting cumulative reward is treated as a stochastic value estimate.
Return a tuple with an uniform policy and the stochastic value estimate for the given
environment `env`.

# Example
```jldoctest
julia> using RLZero, ReinforcementLearningEnvironments
julia> using Random: MersenneTwister

julia> env = RandomWalk1D()
[...]

julia> rollout_oracle = RolloutOracle(MersenneTwister(0))
julia> rollout_oracle(env)
([0.5, 0.5], -1.0)
```

See also [`uniform_oracle`](@ref).
"""
struct RolloutOracle{RNG<:AbstractRNG}
    rng::RNG
end

function (oracle::RolloutOracle)(env::AbstractEnv)
    env = copy(env)
    rewards = 0.0
    original_player = current_player(env)
    while !is_terminated(env)
        player = current_player(env)
        a = rand(oracle.rng, legal_action_space(env))
        env(a)
        r = reward(env)
        rewards += player == original_player ? r : -r
    end
    n = length(legal_action_space(env))
    P = ones(n) ./ n
    return P, rewards
end

end
