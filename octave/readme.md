#

## readTrace

Returns the data of state transition capture by the trace file
Returns
 - the subject position in the maze environment
 - the action value estimated at that position
 - the action applied
 - the reward obtained
 - the resulting subject position
 - the action value estimated at the resulting position
 - the action value estimated at inital position after the reinforcement learning

## qlearn 

Returns the analysis of state transition data
Returns
- The state value estimated at the initial position
- The state value estimated at the final position
- The expected action value estimated at initial position after the transition
- The errors on estimation (distance of expetced from estimated)
- The errors on learning (distance of expetced from what learnt)
