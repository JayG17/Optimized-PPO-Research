import envs.robot_to_exit_env
import gymnasium as gym

"""
Phase 0: Minigrid Setup
Basic minigrid implementation to ensure that the minigrid environment is functional
and has all required connections and functions in the env directory to allow for
successful testing in the later phases.
"""
# initialize environment with render_mode set to human for visual display
env = gym.make('MiniGrid-RobotToExit-v0', render_mode='human')

# set environment params to the values in my env file and begin render
env.reset()
env.render() # render updates the visual display


for _ in range(100):
    # select an action from the sample space
    action = env.action_space.sample()
    # exectue the action and retrieve corresponding data
    data_obs, data_reward_value, data_done, data_trunc, data_info  = env.step(action)
    env.render() # update the visual display
    # if action resulted in a completed state: reached goal (data_done) or max steps
    # used or another case forcing truncation (data_trunc), then end movement 
    if data_done or data_trunc:
        break

env.close() # end environment