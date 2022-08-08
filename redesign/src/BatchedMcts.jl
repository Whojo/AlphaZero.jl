"""
    BatchedMcts

A batched implementation of MCTS that can run on CPU or GPU.

Because this implementation is batched, it is optimized for running MCTS on a large number
of environment instances in parallel. In particular, this implementation is not suitable for
running a large number of MCTS simulations on a single environment quickly (which would
probably require an asynchronous MCTS implementation with virtual loss).

All MCTS trees are represented with a single structure of fixed-sized arrays. This structure
can be hosted either on CPU memory or GPU memory.

In addition, this implementation is not tied to a particular environment interface (such as
`ReinforcementLearningBase` or `CommonRLInterface`). Instead, it relies on an external
*environment oracle* (see `EnvOracle`) that simulates the environment and evaluates states.
This is particularly useful to implement MuZero, in which case the full environment oracle
is implemented by a neural network (in contrast with AlphaZero where a ground-truth
simulator is available).


# Characteristics and limitations

- This implementation supports deterministic, two-player zero-sum games with or without
  intermediate rewards. Support for one-player games should be easy to add and will probably
  become available in the future.
- The memory footprint of the MCTS tree is proportional to the number of environment actions
  (e.g. 9 for tictactoe and ~19x19 for Go). Therefore, this implementation may not be
  suitable for environments offering a very large (or unbounded) number of actions of which
  only a small subset is available in every state.
- States are represented explicitly in the search tree (in contrast with the `SimpleMcts`
  implementation). This increases the memory footprint but avoids simulating the same
  environment transition multiple times (which is essential for MuZero as doing so requires
  making a costly call to a neural network).


# Usage
The examples bellow assume that you run the following code before:
```jldoctest
julia> using RLZero
julia> using .Tests
```

First we need to create a list of environments from which we would like to find the optimal
action. Let's choose the Tic-Tac-Toe game for our experiment.
```jldoctest
julia> envs = [bitwise_tictactoe_draw(), bitwise_tictactoe_winning()]
```

Here, it's worth noting that we used bitwise versions of our position-specific environments
in `./Tests/Common/BitwiseTicTacToe.jl`. Those environments are here to ease the
experimentations and the tests of the package. Bitwise versions are sometimes necessary to
complies with GPU constraints. But in this case, the only motivation to choose them was the
compatibility it offers with `UniformTicTacToeEnvOracle`.

In fact, any environments can be used in `BatchedMcts` if we provide the appropriate
environment oracle. See `EnvOracle` for more details on this.
    
We should then provide a `Policy` to the Mcts. There are most noticeably two arguments to
provide: `device` and `oracle`. The `device` specifies where the algorithm should run. Do
you want it to run on the `CPU` or on the `GPU` ? It's straightforward. The `oracle`
arguments is an `EnvOracle`. You can use the default provided one,
`UniformTicTacToeEnvOracle` or create your own one for other games. For the latter, do not
hesitate to check `EnvOracle` and `check_oracle`.

The `Policy` also accepts other arguments. Refers to the corresponding section to know more.
```jldoctest
julia> policy = BatchedMcts.Policy(;
    device=GPU(),
    oracle=BatchedMcts.UniformTicTacToeEnvOracle()
)
```

After those 2 simple steps we can now call the `explore` function to find out the optimal
action to choose. In the context of AlphaZero/ MuZero, 2 possibilities are offered to you:
`explore` and `gumbel_explore`. Each of them is adapted to a specific context.
- `gumbel_explore` is more suited for the training context of AlphaZero/ MuZero as it
encourages to explore sligthly sub-optimal actions.
- `explore`, on the other hand, is more suited for the inference context of AlphaZero/
MuZero.

Therefore, if you are only interested in the optimal action, always use the `explore`
function.
```jldoctest
julia> tree = BatchedMcts.explore(policy, envs)
```

If you are interested in the exploration undergone, you can check the `Tree` structure.
Otherwise, a simple call to `completed_qvalues` will give you a comprehensive score of how
good each action is. The higher the better of course. We can then use the `argmax` utility
to pickup the best action.
```jldoctest
julia> function get_completed_qvalues(tree)
           ROOT = 1
           (; A, N, B) = BatchedMcts.size(tree)
           tree_size = (Val(A), Val(N), Val(B))

           return [
               BatchedMcts.completed_qvalues(tree, ROOT, bid, tree_size)
               for bid in 1:B
           ]
       end
julia> qs = get_completed_qvalues(tree)
julia> argmax.(qs) # The optimal action for each environment
```

This implementation of batched Mcts tries to provide flexible interfaces to run code on any
devices. You can easily run the tree search on `CPU` or `GPU` with the `device` argument
of `Policy`. If you want the state evaluation or the environment simulation to run on GPU,
you will need to handle it in the `EnvOracle` definition.

By default, `UniformTicTacToeEnvOracle`'s `transition_fn` runs on both `CPU` and `GPU`
depending on the array type of `envs` (a.k.a GPU's `CuArray` vs classic CPU's `Array`). To
write your own custom state evaluation or environment simulation on the appropriate device,
check `EnvOracle` and its example `UniformTicTacToeEnvOracle`.

TODO: This section should show examples of using the module (using jltest?). Ideally, it
should demonstrate different settings such as:

- An AlphaZero-like setting where everything runs on GPU.
- An AlphaZero-like setting where state evaluation runs on GPU and all the rest runs on CPU
  (tree search and environment simulation).
- A MuZero-like setting where everything runs on GPU.

This section should also demonstrate the use of the `check_policy` function to perform
sanity checks on user environments.


# References

- Reference on the Gumbel MCTS extension:
  https://www.deepmind.com/publications/policy-improvement-by-planning-with-gumbel
"""
module BatchedMcts

using Adapt: @adapt_structure
using Base: @kwdef, size
using Distributions: Gumbel
using Random: AbstractRNG
import Base.Iterators.map as imap
using StaticArrays
using CUDA: @allowscalar
using EllipsisNotation

using ..BatchedEnvs
using ..Util.Devices
using ..Util.Devices.KernelFuns: sum, argmax, maximum, softmax

include("./Tests/Common/BitwiseTicTacToe.jl")
using .BitwiseTicTacToe

export EnvOracle, check_oracle, UniformTicTacToeEnvOracle
export Policy, Tree, explore

# # Environment oracles

"""
    EnvOracle(; init_fn, transition_fn)

An environment oracle is defined by two functions: `init_fn` and `transition_fn`. These
functions operate on batches of states directly, enabling efficient parallelization on GPU,
CPU or both.

# The `init_fn` function

`init_fn` takes a vector of environment objects as an argument. Environment objects are of
the same type than those passed to the `explore` and `gumbel_explore` functions. The
`init_fn` function returns a named-tuple of same-size arrays with the following fields:

- `internal_states`: internal representations of the environment states as used by MCTS and
    manipulated by `transition_fn`. `internal_states` must be a single or multi-dimensional
    array whose last dimension is a batch dimension (see examples below). `internal_states`
    must be `isbits` if it is desired to run `BatchedMcts` on `GPU`.
- `valid_actions`: a vector of booleans with dimensions `num_actions` and `batch_id`
  indicating which actions are valid to take (this is disregarded in MuZero).
- `policy_prior`: the policy prior for each states as an `AbstractArray{Float32,2}` with
    dimensions `num_actions` and `batch_id`.
- `value_prior`: the value prior for each state as an `AbstractVector{Float32}`.

# The `transition_fn` function

`transition_fn` takes as arguments a vector of internal states (as returned by `init_fn`)
along with a vector of action ids. Action ids consist in integers between 1 and
`num_actions` and are valid indices for `policy_priors` and `value_priors`. 
    
Note that `init_fn` will always receive the same array than the one passed to `explore` or
`gumbel_explore` as `envs` (which should be a CPU `Array`). But it's a bit more tricky for
`transition_fn`. It may receive both CPU `Array` or GPU `CuArray` depending on the device
specified in `Policy`. To handle both more easily give a look at `Util.Devices` and how it 
is used in `UniformTicTacToeEnvOracle`.

In the context of a `Policy` on the GPU, `transition_fn` can both return a CPU `Array` or a
GPU `CuArray`. The `CuArray` is more adapted as it will prevent a memory transfers, but both
works.

The `transition_fn` function returns a named-tuple of arrays:

- `internal_states`: new states reached after executing the proposed actions (see
    `init_fn`).
- `rewards`: vector of `Float32` indicating the intermediate rewards collected during the
    transitions.
- `terminal`: vector of booleans indicating whether or not the reached states are terminal
    (this is always `false` in MuZero).
- `player_switched`: vector of booleans indicating whether or not the current player
    switched during the transition (always `true` in many board games).
- `valid_actions`, `policy_prior`, `value_prior`: same as for `init_fn`.


# Examples of internal state encodings

- When using AlphaZero on board games such as tictactoe, connect-four, Chess or Go, internal
  states are exact state encodings (since AlphaZero can access an exact simulator). The
  `internal_states` field can be made to have type `AbstractArray{Float32, 4}` with
  dimensions `player_channel`, `board_width`, `board_height` and `batch_id`. Alternatively,
  one can use a one-dimensional vector with element type `State` where
  `Base.isbitstype(State)`. The latter representation may be easier to work with when
  broadcasting non-batched environment implementations on GPU (see
  `Tests.Common.BitwiseTicTacToe.BitwiseTicTacToe` for example).
- When using MuZero, the `internal_states` field typically has type `AbstractArray{Float32,
  2}` where the first dimension corresponds to the size of latent states and the second
  dimension is the batch dimension.

See also [`check_oracle`](@ref), [`UniformTicTacToeEnvOracle`](@ref)
"""
@kwdef struct EnvOracle{I<:Function,T<:Function}
    init_fn::I
    transition_fn::T
end

"""
    check_keys(keys, ref_keys)

Check that the two lists of symbols, `keys` and `ref_keys`, are identical.

Small utilities used  in `check_oracle` to compare keys of named-tuple.
"""
function check_keys(keys, ref_keys)
    return Set(keys) == Set(ref_keys)
end

"""
    check_oracle(::EnvOracle, env)

This function performs some sanity checks to see if an environment oracle is correctly
specified on a given environment instance.

A list of environments `envs` must be specified, along with the `EnvOracle` to check.

The function returns `nothing` if no problems are detected. Otherwise, helpful error
messages are raised. More precisely, `check_oracle` verifies the keys of the returned
named-tuples from `init_fn` an `transition_fn` and the types and dimensions of their lists.

See also [`EnvOracle`](@ref)
"""
function check_oracle(oracle::EnvOracle, envs)
    B = length(envs)

    init_res = oracle.init_fn(envs)
    # Named-tuple check
    @assert check_keys(
        keys(init_res), (:internal_states, :valid_actions, :policy_prior, :value_prior)
    ) "The `EnvOracle`'s `init_fn` function should returned a named-tuple with the " *
        "following fields: internal_states, valid_actions, policy_prior, " *
        "value_prior."

    # Type and dimensions check
    size_valid_actions = size(init_res.valid_actions)
    A, _ = size_valid_actions
    @assert (size_valid_actions == (A, B) && eltype(init_res.valid_actions) == Bool) "The " *
        "`init_fn`'s function should return a `valid_actions` vector with dimensions " *
        "`num_actions` and `batch_id`, and of type `Bool`."
    size_policy_prior = size(init_res.policy_prior)
    @assert (size_policy_prior == (A, B) && eltype(init_res.policy_prior) == Float32) "The " *
        "`init_fn`'s function should return a `policy_prior` vector with dimensions " *
        "`num_actions` and `batch_id`, and of type `Float32`."
    @assert (length(init_res.value_prior) == B && eltype(init_res.value_prior) == Float32) "The " *
        "`init_fn`'s function should return a `value_policy` vector of length " *
        "`batch_id`, and of type `Float32`."

    aids = [
        findfirst(init_res.valid_actions[:, bid]) for
        bid in 1:B if any(init_res.valid_actions[:, bid])
    ]
    envs = [env for (bid, env) in enumerate(envs) if any(init_res.valid_actions[:, bid])]

    transition_res = oracle.transition_fn(envs, aids)
    # Named-tuple check
    @assert check_keys(
        keys(transition_res),
        (
            :internal_states,
            :rewards,
            :terminal,
            :valid_actions,
            :player_switched,
            :policy_prior,
            :value_prior,
        ),
    ) "The `EnvOracle`'s `transition_fn` function should returned a named-tuple with the " *
        "following fields: internal_states, rewards, terminal, valid_actions, " *
        "player_switched, policy_prior, value_prior."

    # Type and dimensions check
    @assert (
        length(transition_res.rewards) == B && eltype(transition_res.rewards) == Float32
    ) "The `transition_fn`'s function should return a `rewards` vector of length " *
        "`batch_id` and of type `Float32`."
    @assert (
        length(transition_res.terminal) == B && eltype(transition_res.terminal) == Bool
    ) "The `transition_fn`'s function should return a `terminal` vector of length " *
        "`batch_id` and of type `Bool`."
    size_valid_actions = size(transition_res.valid_actions)
    @assert (size_valid_actions == (A, B) && eltype(transition_res.valid_actions) == Bool) "The `" *
        "transition_fn`'s function should return a `valid_actions` vector with " *
        "dimensions `num_actions` and `batch_id`, and of type `Bool`."
    @assert (
        length(transition_res.player_switched) == B &&
        eltype(transition_res.player_switched) == Bool
    ) "The `transition_fn`'s function should return a `player_switched` vector of length " *
        "`batch_id`, and of type `Bool`."
    size_policy_prior = size(transition_res.policy_prior)
    @assert (size_policy_prior == (A, B) && eltype(transition_res.policy_prior) == Float32) "The " *
        "`transition_fn`'s function should return a `policy_prior` vector with " *
        "dimensions `num_actions` and `batch_id`, and of type `Float32`."
    @assert (
        length(transition_res.value_prior) == B &&
        eltype(transition_res.value_prior) == Float32
    ) "The `transition_fn`'s function should return a `value_policy` vector of length " *
        "`batch_id`, and of type `Float32`."

    return nothing
end

# ## Example Environment Oracle
# ### Tic-Tac-Toe Environment Oracle
"""
    UniformTicTacToeEnvOracle()

Define an `EnvOracle` object with a uniform policy for the game of Tic-Tac-Toe.

This oracle environment is a wrapper around the BitwiseTicTacToeEnv.
Checkout `./Tests/Common/BitwiseTicTacToe.jl`.

It can be both used on `CPU` & `GPU`.

It was inspired by the RL.jl library. For more details, checkout their documentation:
https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.TicTacToeEnv

See also [`EnvOracle`](@ref)
"""
function UniformTicTacToeEnvOracle()
    get_policy_prior(A, B) = ones(Float32, (A, B)) / A
    get_value_prior(B) = zeros(Float32, B)

    function get_valid_actions(A, B, envs)
        CPU_envs = copy_to_CPU(envs)
        valid_actions = zeros(Bool, (A, B))
        for (bid, env) in enumerate(CPU_envs)
            valid_actions[:, bid] = map(i -> valid_action(env, i), 1:A)
        end
        return valid_actions
    end

    function init_fn(envs)
        A = num_actions(envs[1])
        B = length(envs)

        @assert B > 0
        @assert all(e -> num_actions(e) == A, envs)

        return (;
            internal_states=envs,
            valid_actions=get_valid_actions(A, B, envs),
            policy_prior=get_policy_prior(A, B),
            value_prior=get_value_prior(B),
        )
    end

    # Utilities to access elements returned by `act` in `transition_fn`.
    # `act` return a tuple with the following form:
    #   (state, (; reward, switched))
    get_state(info) = first(info)
    get_reward(info) = last(info).reward
    get_switched(info) = last(info).switched

    function transition_fn(envs, aids)
        A = @allowscalar num_actions(envs[1])
        B = length(envs)

        @assert all(valid_action.(envs, aids)) "Tried to play an illegal move"

        act_info = act.(envs, aids)
        player_switched = get_switched.(act_info)
        rewards = Float32.(get_reward.(act_info))
        internal_states = get_state.(act_info)

        return (;
            internal_states,
            rewards,
            terminal=terminated.(internal_states),
            valid_actions=get_valid_actions(A, B, internal_states),
            player_switched,
            policy_prior=get_policy_prior(A, B),
            value_prior=get_value_prior(B),
        )
    end
    return EnvOracle(; init_fn, transition_fn)
end

# # Policy definition

"""
    Policy{Device, Oracle<:EnvOracle}(; 
        device::Device
        oracle::Oracle
        num_simulations::Int = 64
        num_considered_actions::Int = 8
        value_scale::Float32 = 0.1f0
        max_visit_init::Int = 50
    )

A batch, device-specific MCTS Policy that leverages an external `EnvOracle`.

# Keyword Arguments

- `device::Device`: device on which the policy should preferably run (i.e. `CPU` or `GPU`).
- `oracle::Oracle`: environment oracle handling the environment simulation and the state
    evaluation.
- `num_simulations::Int = 64`: number of simulations to run on the given Mcts `Tree`.
- `num_considered_actions::Int = 8`: number of actions considered by gumbel during
    exploration. Only the `num_conidered_actions` actions with the highest `value_prior`
    probabilities and gumbel noise will be used. It should be a power of 2.
- `value_scale::Float32 = 0.1f0`: multiplying coefficient to weight the qvalues against the
    prior probabilities during exploration. Prior probabilities have, by default, a
    decreasing weight when the number of visits increase.
- `max_visit_init::Int = 50`: artificial increase of the number of visits to weight qvalue
    against prior probabilities on low visit count.

# Notes
The attributes `num_conidered_actions`, `value_scale` and `max_visit_init` are specific to
the gumbel implementation.
"""
@kwdef struct Policy{Device,Oracle<:EnvOracle}
    device::Device
    oracle::Oracle
    num_simulations::Int = 64
    num_considered_actions::Int = 8
    value_scale::Float32 = 0.1f0
    max_visit_init::Int = 50
end

# # Tree datastructure

## Value stored in `tree.parent` for nodes with no parents.
const NO_PARENT = Int16(0)
## Value stored in `tree.children` for unvisited children.
const UNVISITED = Int16(0)
## Value used in various tree's attributes to access root information.
const ROOT = Int16(1)
## Valud used when no action is selected.
const NO_ACTION = Int16(0)

"""
A batch of MCTS trees, represented as a structure of arrays.

# Fields

We provide shape information between parentheses: `B` denotes the batch size, `N` the
maximum number of nodes (i.e. number of simulations) and `A` the number of actions.

## Tree structure and statistics

- `parent`: the id of the parent node or `NO_PARENT` (N, B).
- `num_visits`: number of times the node was visited (N, B).
- `total_values`: the sum of all values backpropagated to a node (N, B).
- `children`: node id of all children or `UNVISITED` for unvisited actions (A, N, B).

## Cached static information

All these fields are used to store the results of calling the environment oracle.

- `state`: state vector or embedding as returned by `init_fn` of the `EnvOracle` (..., N, B).
- `terminal`: whether a node is a terminal node (N, B).
- `valid_actions`: whether or not each action is valid or not (A, N, B).
- `prev_action`: the id of the action leading to this node or 0 (N, B).
- `prev_reward`: the immediate reward obtained when transitioning from the parent from the
   perspective of the parent's player (N, B).
- `prev_switched`: the immediate reward obtained when transitioning from the parent from the
   perspective of the parent's player (N, B).
- `policy_prior`: as given by the `EnvOracle` (A, N, B).
- `value_prior`: as given by the `EnvOracle` (N, B).

# Remarks

- The `Tree` structure is parametric in its field array types since those could be
  instantiated on CPU or GPU (e.g. `Array{Bool, 3}` or 
  `CuArray{Bool, 1, CUDA.Mem.DeviceBuffer}` for `BoolActionArray`). See `create_tree` for
  more details on how a `Tree` is created.
- It is yet to be determined whether a batch of MCTS trees is more cache-friendly when
  represented as a structure of arrays (as is the case here) or as an array of structures
  (as in the `BatchedMctsAos` implementation).
- It is yet to be determined whether or not permuting the `N` and `B` dimensions of all
  arrays would be more cache efficient. An `(N, B)` representation is more conventional, it
  is used in Deepmind's MCTX library (and might be imposed by JAX) and it may provide better
  temporal locality since each thread is looking at a different batch. On the other hand, a
  `(B, N)` layout may provide better spatial locality when copying the results of the
  environment oracle and possibly when navigating trees.
- To complete the previous point, keep in mind that Julia is column-major (compared to the
  row-major, more classical paradigm, in most programming langage like Python). This has the
  noticeable importance that first dimension of a Matrix is continuous. The order of
  dimensions are then reversed compared to a Python implementation (like MCTX) to keep the
  same cache locality.
"""
@kwdef struct Tree{
    StateNodeArray,
    BoolNodeArray,
    Int16NodeArray,
    Float32NodeArray,
    BoolActionArray,
    Int16ActionArray,
    Float32ActionArray,
}
    ## Dynamic stats
    parent::Int16NodeArray
    num_visits::Int16NodeArray
    total_values::Float32NodeArray
    children::Int16ActionArray
    ## Cached oracle info
    state::StateNodeArray
    terminal::BoolNodeArray
    valid_actions::BoolActionArray
    prev_action::Int16NodeArray
    prev_reward::Float32NodeArray
    prev_switched::BoolNodeArray
    policy_prior::Float32ActionArray
    value_prior::Float32NodeArray
end

## https://cuda.juliagpu.org/stable/tutorials/custom_structs/
@adapt_structure Tree

"""
    validate_prior(policy_prior, valid_actions)
    
Correct `policy_prior` to ignore in`valid_actions`.

More precisely, `policy_prior`  that are in`valid_actions` are set to 0. The rest of the
`policy_prior` (which are valid actions) are then l1-normalised.
"""
function validate_prior(policy_prior, valid_actions)
    prior = map(zip(policy_prior, valid_actions)) do (prior, is_valid)
        (is_valid) ? prior : Float32(0)
    end
    prior_sum = mapslices(prior; dims=1) do prior_slice
        sum(prior_slice; init=Float32(0))
    end
    @assert any(prior_sum .!= Float32(0)) "No available actions"
    return prior ./ prior_sum
end

"""
    dims(arr::AbstractArray)
    dims(_)

Return the dimensions of an object.

This utility is used inside `create_tree` so that non-array object have no dimension, rather
than popping an error as `size` do.
"""
dims(arr::AbstractArray) = size(arr)
dims(_) = ()

"""
    create_tree(mcts, envs)
    
Create a `Tree`.

Note that the `ROOT` of the `Tree` are considered explored, as a call to `init_fn` is done
on them. Moreover, `policy_prior` are corrected as specified in `validate_prior`.

See [`Tree`](@ref) for more details.
"""
function create_tree(mcts, envs)
    @assert length(envs) != 0 "There should be at least environment"

    info = mcts.oracle.init_fn(envs)
    A, N, B = size(info.policy_prior)[1], mcts.num_simulations, length(envs)

    num_visits = fill(UNVISITED, mcts.device, (N, B))
    num_visits[ROOT, :] .= 1
    internal_states = DeviceArray(mcts.device){eltype(info.internal_states)}(
        undef, (dims(info.internal_states[1])..., N, B)
    )
    internal_states[.., ROOT, :] = info.internal_states
    valid_actions = fill(false, mcts.device, (A, N, B))
    valid_actions[:, ROOT, :] = info.valid_actions
    policy_prior = zeros(Float32, mcts.device, (A, N, B))
    policy_prior[:, ROOT, :] = validate_prior(info.policy_prior, info.valid_actions)
    value_prior = zeros(Float32, mcts.device, (N, B))
    value_prior[ROOT, :] = info.value_prior

    return Tree(;
        parent=fill(NO_PARENT, mcts.device, (N, B)),
        num_visits,
        total_values=zeros(Float32, mcts.device, (N, B)),
        children=fill(UNVISITED, mcts.device, (A, N, B)),
        state=internal_states,
        terminal=fill(false, mcts.device, (N, B)),
        valid_actions,
        prev_action=zeros(Int16, mcts.device, (N, B)),
        prev_reward=zeros(Float32, mcts.device, (N, B)),
        prev_switched=fill(false, mcts.device, (N, B)),
        policy_prior,
        value_prior,
    )
end

"""
    Base.size(tree::Tree)

Return the number of actions (`A`), the number of simulations (`N`) and the number of
environments in the batch (`B`) of a `tree` as named-tuple `(; A, N, B)`.
"""
function Base.size(tree::Tree)
    A, N, B = size(tree.children)
    return (; A, N, B)
end

"""
    batch_size(tree)

Return the number of environments in the batch of a `tree`.
"""
batch_size(tree) = size(tree).B

# # MCTS implementation

# ## Basic MCTS functions

"""
    value(tree, cid, bid)

Return the absolute value of game position.

The formula for a given node is:
    $$ (prior_value + total_rewards) / num_visits $$

With `prior_value` the value as estimated by the oracle, `total_rewards` the sum of rewards
obtained from episodes including this node during exploration and `num_visits` the number of
episodes including this node.

See also [`qvalue`](@ref)
"""
value(tree, cid, bid) = tree.total_values[cid, bid] / tree.num_visits[cid, bid]

"""
    qvalue(tree, cid, bid)

Return the value of game position from the perspective of its parent node.
I.e. a good position for your opponent is bad one for you.

See also [`value`](@ref)
"""
qvalue(tree, cid, bid) = value(tree, cid, bid) * (-1)^tree.prev_switched[cid, bid]

"""
    root_value_estimate(tree, cid, bid, ::Tuple{Val{A},Any,Any}) where {A}

Compute a value estimation of a node at this stage of the exploration.

The estimation is based on its `value_prior`, its number of visits, the `qvalue` of its
children and their associated `policy_prior`.
"""
function root_value_estimate(tree, cid, bid, ::Tuple{Val{A},Any,Any}) where {A}
    total_qvalues = Float32(0)
    total_prior = Float32(0)
    total_visits = UNVISITED
    for aid in 1:A
        cnid = tree.children[aid, cid, bid]
        (cnid == UNVISITED) && continue

        total_qvalues += tree.policy_prior[aid, cid, bid] * qvalue(tree, cnid, bid)
        total_prior += tree.policy_prior[aid, cid, bid]
        total_visits += tree.num_visits[cnid, bid]
    end
    children_value = total_qvalues
    total_prior > 0 && (children_value /= total_prior)
    return (tree.value_prior[cid, bid] + total_visits * children_value) / (1 + total_visits)
end

"""
    completed_qvalues(tree, cid, bid, tree_size::Tuple{Val{A},Any,Any}) where {A}
    
Return a list of estimated qvalue of all children for a given node.

More precisely, if its child have been visited at least one time, it computes its real
`qvalue`, otherwise, it uses the `root_value_estimate` of node instead.
"""
function completed_qvalues(tree, cid, bid, tree_size::Tuple{Val{A},Any,Any}) where {A}
    root_value = root_value_estimate(tree, cid, bid, tree_size)
    ret = imap(1:A) do aid
        (!tree.valid_actions[aid, cid, bid]) && return -Inf32

        cnid = tree.children[aid, cid, bid]
        return cnid != UNVISITED ? qvalue(tree, cnid, bid) : root_value
    end
    return SVector{A}(ret)
end

"""
    get_num_child_visits(tree, cid, bid, ::Tuple{Val{A},Any,Any}) where {A}

Return the number of visits of each children from the given node.
"""
function get_num_child_visits(tree, cid, bid, ::Tuple{Val{A},Any,Any}) where {A}
    ret = imap(1:A) do aid
        cnid = tree.children[aid, cid, bid]
        (cnid != UNVISITED) ? tree.num_visits[cnid, bid] : UNVISITED
    end
    return SVector{A}(ret)
end

"""
    qcoeff(mcts, tree, cid, bid, tree_size)

Compute a gumbel-related ponderation of `qvalue`.

Through time, as the number of visits increase, the influence of `qvalue` builds up
relatively to `policy_prior`.
"""
function qcoeff(mcts, tree, cid, bid, tree_size)
    # XXX: init is necessary for GPUCompiler right now...
    max_child_visit = maximum(
        get_num_child_visits(tree, cid, bid, tree_size); init=UNVISITED
    )
    return mcts.value_scale * (mcts.max_visit_init + max_child_visit)
end

"""
    target_policy(mcts, tree, cid, bid, tree_size::Tuple{Val{A},Any,Any}) where {A}
    
Return the policy of a node.

I.e. a softmaxed-score of how much each actions should be played. The higher, the better.
"""
function target_policy(mcts, tree, cid, bid, tree_size::Tuple{Val{A},Any,Any}) where {A}
    qs = completed_qvalues(tree, cid, bid, tree_size)
    policy = SVector{A}(imap(aid -> tree.policy_prior[aid, cid, bid], 1:A))
    return softmax(log.(policy) + qcoeff(mcts, tree, cid, bid, tree_size) * qs)
end

"""
    select_nonroot_action(mcts, tree, cid, bid, tree_size)

Select the most appropriate action to explore.

`NO_ACTION` is returned if no actions are available.
"""
function select_nonroot_action(mcts, tree, cid, bid, tree_size)
    policy = target_policy(mcts, tree, cid, bid, tree_size)
    num_child_visits = get_num_child_visits(tree, cid, bid, tree_size)
    total_visits = sum(num_child_visits; init=UNVISITED)
    return Int16(
        argmax(
            policy - Float32.(num_child_visits) / (total_visits + 1);
            init=(NO_ACTION, -Inf32),
        ),
    )
end

# ## Core MCTS algorithm
function select(mcts, tree, bid, tree_size; start=ROOT)
    cur = start
    while true
        if tree.terminal[cur, bid]
            # returns current terminal, but no action played
            return cur, NO_ACTION
        end
        aid = select_nonroot_action(mcts, tree, cur, bid, tree_size)
        @assert aid != NO_ACTION

        cnid = tree.children[aid, cur, bid]
        if cnid != UNVISITED
            cur = cnid
        else
            # returns parent and action played
            return cur, aid
        end
    end
    return nothing
end

function eval!(mcts, tree, simnum, parent_frontier)
    B = batch_size(tree)
    # How the parent_frontier's tuples are formed
    action = last
    parent = first

    # Get terminal nodes at `parent_frontier`
    non_terminal_mask = @. action(parent_frontier) != NO_ACTION
    # No new node to expand (a.k.a only terminal node on the frontier)
    (!any(non_terminal_mask)) && return parent.(parent_frontier)

    parent_ids = parent.(parent_frontier[non_terminal_mask])
    action_ids = action.(parent_frontier[non_terminal_mask])
    non_terminal_bids = DeviceArray(mcts.device)(Base.OneTo(B))[non_terminal_mask]

    parent_cartesian_ids = CartesianIndex.(parent_ids, non_terminal_bids)
    parent_states = tree.state[.., parent_cartesian_ids]
    info = mcts.oracle.transition_fn(parent_states, action_ids)

    # Create nodes and save `info`
    children_cartesian_ids = CartesianIndex.(action_ids, parent_ids, non_terminal_bids)

    tree.parent[simnum, non_terminal_mask] = parent_ids
    tree.children[children_cartesian_ids] .= simnum
    tree.state[.., simnum, non_terminal_mask] = info.internal_states
    tree.terminal[simnum, non_terminal_mask] = info.terminal
    tree.valid_actions[:, simnum, non_terminal_mask] = info.valid_actions
    tree.prev_action[simnum, non_terminal_mask] = action_ids
    tree.prev_reward[simnum, non_terminal_mask] = info.rewards
    tree.prev_switched[simnum, non_terminal_mask] = info.player_switched
    tree.policy_prior[:, simnum, non_terminal_mask] = info.policy_prior # TODO: validate_prior
    tree.value_prior[simnum, non_terminal_mask] = info.value_prior

    # Update frontier
    frontier = parent.(parent_frontier)
    frontier[non_terminal_mask] .= simnum

    return frontier
end

function select_and_eval!(mcts, tree, simnum)
    (; A, N, B) = size(tree)
    tree_size = (Val(A), Val(N), Val(B))

    batch_indices = DeviceArray(mcts.device)(1:B)
    parent_frontier = map(batch_indices) do bid
        select(mcts, tree, bid, tree_size)
    end

    return eval!(mcts, tree, simnum, parent_frontier)
end

function backpropagate!(mcts, tree, frontier)
    B = batch_size(tree)
    batch_ids = DeviceArray(mcts.device)(1:B)
    map(batch_ids) do bid
        sid = frontier[bid]
        val = tree.value_prior[sid, bid]
        while true
            val += tree.prev_reward[sid, bid]
            (tree.prev_switched[sid, bid]) && (val = -val)
            tree.num_visits[sid, bid] += Int16(1)
            tree.total_values[sid, bid] += val
            if tree.parent[sid, bid] != NO_PARENT
                sid = tree.parent[sid, bid]
            else
                return nothing
            end
        end
    end
    return nothing
end

function explore(mcts, envs)
    tree = create_tree(mcts, envs)
    (; N) = size(tree)
    for simnum in 2:N
        frontier = select_and_eval!(mcts, tree, simnum)
        backpropagate!(mcts, tree, frontier)
    end
    return tree
end

function get_sequence_of_considered_visits(max_num_considered_actions, num_simulations)
    (max_num_considered_actions <= 1) &&
        return SVector{num_simulations,Int16}(0:(num_simulations - 1))

    num_halving_steps = Int(ceil(log2(max_num_considered_actions)))
    sequence = Int16[]
    visits = zeros(Int16, max_num_considered_actions)

    num_considered = max_num_considered_actions
    while length(sequence) < num_simulations
        num_extra_visits = max(1, num_simulations ÷ (num_halving_steps * num_considered))
        for _ in 1:num_extra_visits
            append!(sequence, visits[1:num_considered])
            visits[1:num_considered] .+= 1
        end
        num_considered = max(2, num_considered ÷ 2)
    end

    return SVector{num_simulations}(sequence[1:num_simulations])
end

function get_table_of_considered_visits(mcts, ::Tuple{Val{A},Any,Any}) where {A}
    ret = imap(1:A) do num_considered_actions
        get_sequence_of_considered_visits(num_considered_actions, mcts.num_simulations)
    end
    return SVector{A}(ret)
end

function gumbel_select_root(
    mcts,
    tree,
    bid,
    gumbel,
    table_of_considered_visits,
    child_total_visits,
    tree_size::Tuple{Val{A},Any,Val{B}},
) where {A,B}
    num_valid_actions = sum(aid -> tree.valid_actions[aid, ROOT, bid], 1:A; init=NO_ACTION)
    num_considered = min(mcts.num_considered_actions, num_valid_actions)

    num_visits = get_num_child_visits(tree, ROOT, bid, tree_size)
    considered_visits = table_of_considered_visits[num_considered][child_total_visits]
    penality_value = imap(1:A) do aid
        (num_visits[aid] == considered_visits) ? Float32(0) : -Inf32
    end
    penality = SVector{A}(penality_value)

    qs = completed_qvalues(tree, ROOT, bid, tree_size)
    norm_qs = qs .* qcoeff(mcts, tree, ROOT, bid, tree_size)
    policy = SVector{A}(imap(aid -> tree.policy_prior[aid, ROOT, bid], 1:A))
    batch_gumbel = SVector{A}(imap(aid -> gumbel[aid, bid], 1:A))
    scores = batch_gumbel + log.(policy) + norm_qs + penality
    return Int16(argmax(scores; init=(NO_ACTION, -Inf32)))
end

function gumbel_select_and_eval!(
    mcts, tree, simnum, gumbel, table_of_considered_visits, tree_size::Tuple{Any,Any,Val{B}}
) where {B}
    batch_indices = DeviceArray(mcts.device)(1:B)
    parent_frontier = map(batch_indices) do bid
        aid = gumbel_select_root(
            mcts,
            tree,
            bid,
            gumbel,
            table_of_considered_visits,
            simnum - ROOT,
            tree_size,
        )
        @assert aid != NO_ACTION

        cnid = tree.children[aid, ROOT, bid]
        if (cnid != UNVISITED)
            select(mcts, tree, bid, tree_size; start=cnid)
        else
            (ROOT, aid)
        end
    end

    return eval!(mcts, tree, simnum, parent_frontier)
end

function gumbel_explore(mcts, envs, rng::AbstractRNG)
    tree = create_tree(mcts, envs)
    (; A, B, N) = size(tree)
    tree_size = (Val(A), Val(N), Val(B))

    gumbel = DeviceArray(mcts.device)(rand(rng, Gumbel(), (A, B)))
    table_of_considered_visits = DeviceArray(mcts.device)(
        get_table_of_considered_visits(mcts, tree_size)
    )

    for simnum in 2:N
        frontier = gumbel_select_and_eval!(
            mcts, tree, simnum, gumbel, table_of_considered_visits, tree_size
        )
        backpropagate!(mcts, tree, frontier)
    end
    return tree
end

end