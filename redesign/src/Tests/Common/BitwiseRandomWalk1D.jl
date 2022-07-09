module BitwiseRandomWalk1D

using Setfield

using ....BatchedEnvs

export BitwiseRandomWalk1DEnv

"""
A RandomWalk1D Environment.

An agent is placed at the `start_pos` and can move left or right (stride is
defined in `actions`). The game terminates when the agent reaches either end (i.e. 1 or
`length`) and receives a `reward` correspondingly.
"""
Base.@kwdef struct BitwiseRandomWalk1DEnv
    rewards::Pair{Float64,Float64} = -1.0 => 1.0
    actions::Vector{Int} = [-1, 1] # TODO: Do not compile on GPU ?
    length::Int = 7
    pos::Int = length ÷ 2 + 1
end

BatchedEnvs.num_actions(env::BitwiseRandomWalk1DEnv) = length(env.actions)

function Base.show(io::IO, ::MIME"text/plain", env::BitwiseRandomWalk1DEnv)
    for i in 1:(env.length)
        if i == env.pos
            print(io, "X ")
        else
            print(io, "_ ")
        end
    end
end

BatchedEnvs.valid_action(env::BitwiseRandomWalk1DEnv, pos) = pos in 1:length(env.actions)

function get_reward(env::BitwiseRandomWalk1DEnv)
    if env.pos == 1
        first(env.rewards)
    elseif env.pos == env.length
        last(env.rewards)
    else
        0
    end
end

function BatchedEnvs.act(env::BitwiseRandomWalk1DEnv, action)
    new_env = @set env.pos = clamp(env.pos + env.actions[action], 1, env.length)
    return new_env, (; reward=get_reward(new_env), switched=false)
end

function BatchedEnvs.act(env::BitwiseRandomWalk1DEnv, action_list::AbstractArray)
    new_env = env
    for action in action_list
        new_env, _ = BatchedEnvs.act(new_env, action)
    end
    return new_env, (; reward=get_reward(new_env), switched=false)
end

function BatchedEnvs.terminated(env::BitwiseRandomWalk1DEnv)
    return env.pos == 1 || env.pos == env.length
end

end