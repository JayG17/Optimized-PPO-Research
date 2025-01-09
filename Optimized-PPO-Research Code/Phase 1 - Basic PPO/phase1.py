import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.robot_to_exit_env import RobotToExitEnv  # Ensure the environment is correctly imported
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import time
"""
Phase 1: Basic PPO
Basic PPO implementation in the minigrid environment for intial benchmarks.
"""

# initialize environment using DummyVecEnv since stable_baselines3 libary 
# requires the vectorized environment form. DummyVecEnv is a wrapper for the
# environment that basiclly makes it easier for RL testing with by making 
# it can be easy to reset, duplicate, batch process and more.
env = DummyVecEnv([lambda: gym.make('MiniGrid-RobotToExit-v0')])

# normalize the environment observations and rewards, to help stabilize trianing
# , and observation clip to a +- 10 to avoid outliers.
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)


# initialize the basic PPO model
# https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
model = PPO(
        policy = "MlpPolicy", # use stablebaselines multi-layer perceptron policy for the model decision-making
        env = env, # specify the environment where the agent performs
        verbose = 1, # set verbosity level: 0 for no output, 1 for the standard logger outputs
        learning_rate = 1e-4, # set the rate in which the model updates weight values, with 1e-4 as a smaller, more basic and stable rate.
        n_steps = 2048, # number of steps to collect per batch, before the model updates
        clip_range = 0.2, # set a clip for helping to stabilize policy updates and overall learning
        gamma = 0.99, # set the gamma discount factor value, placing emphasis on gamma=0.99 (prioritize future rewards more than immediate)
        ent_coef = 0.01 # set entropy coefficient to a lower value 0.01 (lower penalty for certainty, prioritizes stability and exploitation over exploration)
)

# record start time
start_time = time.time()

# train model for 100,000 steps
model.learn(total_timesteps=100000)

# record end time
end_time = time.time()

# evaluate total training time
training_time = end_time - start_time

# save the model as .zip and environment statistics as .pkl for evaluation
model.save("outputs/phase1-model")
env.save("outputs/stats1.pkl")

# evalute model over different episode amounts
episode_amounts = [100, 1000, 5000]
with open("outputs/phase1.txt", "w") as FILEIN:
    FILEIN.write(f"Phase 1 (Basic PPO) training completion time: {training_time:.3f} seconds\n")
    for current_episode_amount in episode_amounts:
        total_rewards, total_steps = 0, 0 # reset total rewards and steps
        for episode in range(current_episode_amount):
            obs = env.reset() # reset environment at start of each episode
            episode_rewards, episode_steps, episode_done = 0, 0, False # reset params at start of each episode
            while not episode_done: # continue until episode end
                action, _states = model.predict(obs) # predict next action based on observation
                obs, new_reward, episode_done, _info = env.step(action) # take action using the environment step
                episode_rewards += new_reward # update the reward accumulated through the episode
                episode_steps += 1 # update the step amount taken this far
            total_rewards += episode_rewards # at episode completion, update total rewards
            total_steps += episode_steps # at episode completion, update total steps
            if episode % 100 == 0: # every 100 episodes, print information about the current episode
                print(f"Episode {(episode+1)}: Reward={episode_rewards}, Steps={episode_steps}")

        # output results to txt
        display_reward_avg = total_rewards/current_episode_amount
        display_step_avg = total_steps/current_episode_amount
        FILEIN.write(f"\nPerformance over {current_episode_amount} episodes:")
        FILEIN.write(f"\nReward Average: [{display_reward_avg}]")
        FILEIN.write(f"\nSteps to Completion Average: {display_step_avg}\n")
