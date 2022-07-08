"""
Interface for batchable environements that can be run on the GPU.

# Implementation
To implement this interface, please provide a method for each of these functions as
described bellow: `num_actions`, `valid_actions`, `act` and `terminated`.

This is just an interface, no fallback method are provided.

Checkout "src/Tests/Common/BitwiseTicTacToe.jl" for an example implementation of this
interface.
"""
module BatchedEnvs

export num_actions, valid_action, act, terminated

"""
    function num_actions(env::BatchedEnvs)

Return the total number of actions for a given environment `env`

This number includes the legal moves as well as the illegal ones.
"""
function num_actions end

"""
    function valid_action(env::BatchedEnvs, pos)

Return boolean indicating whether the action at index `pos` is legal in the
environment `env`.
"""
function valid_action end

"""
    function act(env::BatchedEnvs, pos)

Play the action `pos` on the current environement `env`.

`pos` is an index in the actions list. Return the new environement and a nammed tupple with
the acquired `reward` and whether the player has `switched` or not.

    function act(env::BatchedEnvs, pos_list::AbstractArray)

Play iterativelly all actions from `pos_list` on the current environement `env`.

Return the new environement and a nammed tupple with the acquired `reward` and whether the
player has `switched` or not. Useful to setup a game starting position.
"""
function act end

"""
    function terminated(env::BatchedEnvs)
  
Return a boolean indicating whether the environement `env` has terminated.

i.e. that the game was won by either of the players or ended up in a draw in most game.
"""
function terminated end

end
