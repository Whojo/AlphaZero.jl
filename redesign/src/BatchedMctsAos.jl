"""
An batched MCTS implementation that can run on GPU where trees
are represented in Array of Structs format.
"""
module BatchedMctsAos

using StaticArrays
using Distributions: sample, Gumbel
using Random: AbstractRNG
using Base: @kwdef
using Setfield
using Base.Iterators: map as imap

using ..BatchedEnvs
using ..Util.Devices
using ..Util.Devices.KernelFuns: sum, argmax, maximum, softmax

"""
A batch, device-specific MCTS Policy that leverages an external oracle.

# Attributes
- `oracle::Oracle`: function that, given an environment (as defined in
    ReinforcementLearningEnvironment.jl), computes the prior probabilities associated with
    its actions (children of a Mcts `Tree`) and the value associated with the provided
    environment.
- `device::Device`: device on which the policy should preferably run (i.e. CPU or GPU).
- `num_simulations::Int = 64`: number of simulations to run on the given Mcts `Tree`.
- `num_considered_actions::Int = 8`: ... # TODO
- `value_scale::Float32 = 0.1f0`: ... # TODO
- `max_visit_init::Int = 50`: ... # TODO

# Notes
The attributes `num_conidered_actions`, `value_scale` and `max_visit_init` are specific to
the gumbel implementation.
"""
@kwdef struct Policy{Oracle,Device}
    oracle::Oracle
    device::Device
    num_simulations::Int = 64
    num_considered_actions::Int = 8
    value_scale::Float32 = 0.1f0
    max_visit_init::Int = 50
end

"""
An MCTS tree.

# Attributes
- `state::State`: current state (a.k.a environment) for this `Node`.
- `parent::Int16 = Int16(0)`: direct parent of this `Node`. This argument is used for
     backtracking.
- `prev_action::Int16 = Int16(0)`: store the index of the last action played that leads to
     the current state.
- `prev_reward::Float32 = 0.0f0`: store the reward acquired with the last action played that
     leads to the current state.
- `prev_switched::Bool = false`: store whether the player turn to play has switched with the
     last action played that leads to the current state.
- `terminal::Bool = false`: whether the actual `state` is a terminal `Node`. i.e. that one
    player won or that the game ended up in a draw.
- `valid_actions::SVector{NumActions,Bool} = @SVector zeros(Bool, NumActions)
- `prior::SVector{NumActions,Float32} = @SVector zeros(Float32, NumActions)`: prior
     probabilities estimated by the oracle (e.g. `uniform_oracle`) when a `Tree` is created.
- `value_prior::Float32 = 0.0f0`: value estimated by the oracle (e.g. `uniform_oracle`) when a
     `Tree` is created.
- `children::SVector{NumActions,Int16} = @SVector zeros(Int16, NumActions)`: children of
    the current `Tree`. The indexes of each child correspond to the one given by
    `legal_action_space(env)`, with `env` the
    implicit environment (as defined in ReinforcementLearningEnvironment.jl) associated
    with this `Tree` during exploration (i.e. `explore` function).
- `total_rewards::Float32 = 0.0f0`: sum of the accumulated rewards of the current 
    `Tree` during the Mcts exploration (i.e. `explore` function).
- `num_visits::Int16 = Int16(1)`: number of visits of the current `Tree` during the Mcts
    exploration (i.e. `explore` function).

# Notes
This Tree is represented as an Array of Structures.
"""
@kwdef struct Node{NumActions,State}
    # Static info created at initialization
    state::State
    parent::Int16 = Int16(0)
    prev_action::Int16 = Int16(0)
    prev_reward::Float32 = 0.0f0
    prev_switched::Bool = false
    terminal::Bool = false
    valid_actions::SVector{NumActions,Bool} = @SVector zeros(Bool, NumActions)
    # Oracle info
    prior::SVector{NumActions,Float32} = @SVector zeros(Float32, NumActions)
    value_prior::Float32 = 0.0f0
    # Dynamic info
    children::SVector{NumActions,Int16} = @SVector zeros(Int16, NumActions)
    total_rewards::Float32 = 0.0f0
    num_visits::Int16 = Int16(1)
end

"""
    function Node{na}(state; args...) where {na}
    
Constructor of a `Node` depending on a `state`.

The other accepted arguments are specified in the documentation section of the `Node` structure.
"""
function Node{na}(state; args...) where {na}
    terminal = terminated(state)
    if terminal
        valid_actions = SVector{na,Bool}(false for _ in 1:na)
    else
        valid_actions = SVector{na,Bool}(false for _ in 1:na)
    end
    return Node{na,typeof(state)}(; state, terminal, valid_actions, args...)
end

"""
    function create_tree(mcts, envs)

Constructor of a `Tree` (which is an Array of Structures) depending on a batch of
environments `envs` and a given `mcts` policy.
"""
function create_tree(mcts, envs)
    env = envs[1]
    na = num_actions(env)
    ne = length(envs)
    ns = mcts.num_simulations
    Arr = DeviceArray(mcts.device)
    # We index tree nodes with (batchnum, simnum)
    # This is unusual but this has better cache locality in this case
    tree = Arr{Node{na,typeof(env)}}(undef, (ne, ns))
    tree[:, 1] = Arr([Node{na}(e) for e in envs])
    return tree
end

const Tree{N,S} = AbstractArray{Node{N,S}}

"""
    function tree_dims(tree::Tree{N,S}) where {N,S}

Return the 3 dimensions of a given `tree`:
- Number of actions
- Number of environments
- Number of simulations
"""
function tree_dims(tree::Tree{N,S}) where {N,S}
    na = N
    ne, ns = size(tree)
    return (; na, ne, ns)
end

"""
    function eval_states!(mcts, tree, frontier)

Evaluate the node at the `frontier` of the `tree` (i.e. the last unexplored or terminal
node reached) from `mcts.oracle`.
"""
function eval_states!(mcts, tree, frontier)
    (; na, ne) = tree_dims(tree)
    prior = (@SVector ones(Float32, na)) / na
    Devices.foreach(1:ne, mcts.device) do batchnum
        nid = frontier[batchnum]
        node = tree[batchnum, nid]
        if !node.terminal
            @set! node.prior = prior
            @set! node.value_prior = 0.0f0
            tree[batchnum, nid] = node
        end
    end
    return nothing
end

"""
    function value(node)
        
Compute the qvalue for the given `node`.
    
This qvalue is not weighted from the player point of view.

See also `qvalue`.
"""
value(node) = node.total_rewards / node.num_visits

"""
    function qvalue(node)

Compute the qvalue for the given `node`.

This qvalue is weighted from the player point of view.

See also `value`.
"""
qvalue(child) = value(child) * (-1)^child.prev_switched

"""
    function root_value_estimate(tree, node, bid)
        
Compute the root value for a given `node`.

`Tree` and the batch id `bid` must be provided to access children of the current `node`.
Its value depends on the `num_visits` and `qvalue` of all of its children combined with its
`oracle_value` and `prior` estimate. See the code for more details.
"""
function root_value_estimate(tree, node, bid)
    total_qvalues = 0.0f0
    total_prior = 0.0f0
    total_visits = 0
    for (i, cnid) in enumerate(node.children)
        if cnid > 0  # if the child was visited
            child = tree[bid, cnid]
            total_qvalues += node.prior[i] * qvalue(child)
            total_prior += node.prior[i]
            total_visits += child.num_visits
        end
    end
    children_value = total_qvalues
    total_prior > 0 && (children_value /= total_prior)
    return (node.value_prior + total_visits * children_value) / (1 + total_visits)
end

"""
    function completed_qvalues(tree, node, bid)

Return a list of estimated qvalue of all children for a given `node`.

More precisely, if its child have been visited at least one time, it computes its real
`qvalue`, otherwise, it uses the `root_value_estimate` of `node` instead.

# Example
```jldoctest
julia> using RLZero
julia> using .Tests

julia> policy = BatchedMctsAosTests.uniform_mcts_tic_tac_toe(CPU())
[...]

julia> envs = BatchedMctsAosTests.tic_tac_toe_winning_envs()
[...]

julia> tree = explore(policy, envs)
[...]

julia> 
"""
function completed_qvalues(tree, node, bid)
    root_value = root_value_estimate(tree, node, bid)
    na = length(node.children)
    ret = imap(1:na) do i
        cnid = node.children[i]
        return cnid > 0 ? value(tree[bid, cnid]) : root_value
    end
    return SVector{na}(ret)
end

function num_child_visits(tree, node, bid, i)
    cnid = node.children[i]
    return cnid > 0 ? tree[bid, cnid].num_visits : Int16(0)
end

function qcoeff(mcts, tree, node, bid)
    na = length(node.children)
    # init is necessary for GPUCompiler right now...
    max_child_visit = maximum(1:na; init=Int16(0)) do i
        num_child_visits(tree, node, bid, i)
    end
    return mcts.value_scale * (mcts.max_visit_init + max_child_visit)
end

function target_policy(mcts, tree, node, bid)
    qs = completed_qvalues(tree, node, bid)
    return softmax(log.(node.prior) + qcoeff(mcts, tree, node, bid) * qs)
end

function select_nonroot_action(mcts, tree, node, bid)
    policy = target_policy(mcts, tree, node, bid)
    na = length(node.children)
    total_visits = sum(i -> num_child_visits(tree, node, bid, i), 1:na; init=0)
    return argmax(1:na; init=(0, -Inf32)) do i
        ratio = Float32(num_child_visits(tree, node, bid, i)) / (total_visits + 1)
        return policy[i] - ratio
    end
end

function select!(mcts, tree, simnum, bid)
    (; na) = tree_dims(tree)
    cur = Int16(1)  # start at the root
    while true
        node = tree[bid, cur]
        if node.terminal
            return cur
        end
        i = select_nonroot_action(mcts, tree, node, bid)
        cnid = node.children[i]
        if cnid > 0
            # The child is already in the tree so we proceed.
            cur = cnid
        else
            # The child is not in the tree so we add it and return.
            newstate, info = act(node.state, i)
            child = Node{na}(
                newstate;
                parent=cur,
                prev_action=i,
                prev_reward=info.reward,
                prev_switched=info.switched,
            )
            tree[bid, simnum] = child
            return Int16(simnum)
        end
    end
end

# Start from the root and add a new frontier (whose index is returned)
# After the node is added, one expect the oracle to be called on all
# frontier nodes where terminal=false.
function select!(mcts, tree, simnum)
    (; ne) = tree_dims(tree)
    batch_ids = DeviceArray(mcts.device)(1:ne)
    frontier = map(batch_ids) do bid
        return select!(mcts, tree, simnum, bid)
    end
    return frontier
end

# Value: if terminal node: terminal value / otherwise: network value
function backpropagate!(mcts, tree, frontier)
    (; ne) = tree_dims(tree)
    batch_ids = DeviceArray(mcts.device)(1:ne)
    map(batch_ids) do bid
        node = tree[bid, frontier[bid]]
        val = node.value_prior
        while true
            @set! node.num_visits += Int16(1)
            @set! node.total_rewards += val
            if node.parent > 0
                (node.prev_switched) && (val = -val)
                val += node.prev_reward
                node = tree[bid, node.parent]
            else
                return nothing
            end
        end
    end
    return nothing
end

function explore(mcts, envs)
    tree = create_tree(mcts, envs)
    (; ne, ns) = tree_dims(tree)
    frontier = DeviceArray(mcts.device)(ones(Int16, ne))
    eval_states!(mcts, tree, frontier)
    for i in 2:ns
        frontier = select!(mcts, tree, i)
        eval_states!(mcts, tree, frontier)
        backpropagate!(mcts, tree, frontier)
    end
    return tree
end

end
