import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.robot_to_exit_env import RobotToExitEnv  # Ensure the environment is correctly imported
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import time
import torch
"""
Phase 4: Enhanced Code-Level Optimizations and OPPO+ Algorithm Merge
Implements an advanced combination of Code-Level Optimizations for PPO with the
OPPO+ algorithm wihtin the MiniGrid environment, aiming to enhance the overall
benchmarking performance.
"""

# initialize environment using DummyVecEnv since stable_baselines3 libary 
# requires the vectorized environment form. DummyVecEnv is a wrapper for the
# environment that basiclly makes it easier for RL testing with by making 
# it can be easy to reset, duplicate, batch process and more.
env = DummyVecEnv([lambda: gym.make('MiniGrid-RobotToExit-v0')])

# normalize the environment observations and rewards, to help stabilize trianing
# , and observation clip to a +- 10 to avoid outliers.
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=5.0)

# custom policy configurations to further develop the code-level optimizations
# for the policy network which maps observations from the env to agent actions.
# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
policy_configs = dict(
    # Orthogonal Initalization:
    # This concept sets weights to be orthogonal/perrpendicular to each other, which 
    # in turn helps to stabilize weight scaling when training. This is used instead
    # of the standard 'random' weight initialization methods like Gaussian distribution
    # which can possibly make weights in random/artibrary directions.
    ortho_init = True,

    # Policy and Value Network Architecture:
    # This concept sets the structure of the neural network used in the policy
    # model. This sets two layers of 64 units for the pi (policy network - that decides
    # actions) and two layers of 64 units for the vf (value function - that evaluates
    # avctions). This new layered structure helps the model learn more complex patterns.
    # Essentially, adding the two layers means the network has two groups of 64 units to
    # process information in each step for recognizing patterns and making decisions.
    net_arch = dict(pi=[64, 64], vf=[64, 64]),

    # Hyperbolic Tan Activations:
    # This concept sets the activation function as the hyperbolic tan (Tanh) function.
    # This scales the output to be between -1 and 1, which is a helpful bound for values
    # and helps to promote nice and stable gradients in the network.
    activation_fn=torch.nn.Tanh
)

# Adam Learning Rate Annealing:
# Basically, Adam (Adaptive Moment Esitmation) Learning Rate Annealing is a concept where
# the learning rate is decreased when training as a way to help the model converge to the
# already learned knowledge and make it more fine-tuned. This is applicable to RL in order
# to make it more emphasized on exploitation.
def anneal_learn_rate_func(progress_remaining):
    min_lr = 1e-5  # Minimum learning rate
    return max(2.5e-4 * (0.5 + 0.5*progress_remaining), min_lr)

# https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
model = PPO(
        policy = "MlpPolicy", # use stablebaselines multi-layer perceptron policy for the model decision-making
        env = env, # specify the environment where the agent performs
        verbose = 1, # set verbosity level: 0 for no output, 1 for the standard logger outputs
        n_steps = 2048, # number of steps to collect per batch, before the model updates
        clip_range = 0.2, # set a clip for helping to stabilize policy updates and overall learning
        gamma = 0.99, # set the gamma discount factor value, placing emphasis on gamma=0.99 (prioritize future rewards more than immediate)
        ent_coef = 0.002, # set reduced entropy coefficient to a lower value 0.002 for balancing with OPPO+ bonus 
        vf_coef = 0.5, # value function coefficient set to 0.5 to balance policy and value learning
        max_grad_norm = 0.5, # gradient clipping to limit max gradient size to ensure stability during large updates
        batch_size = 512, # increased batch size for more stability, and smaller variance in the policy updates throughout
        n_epochs = 10, # increased epochs within each update to allow the model to learn from each batch more times
        policy_kwargs = policy_configs, # implementation of custom policy configurations
        learning_rate = anneal_learn_rate_func, # set the rate in which the model updates weight values, with 1e-4 as a smaller, more basic and stable rate.
)


# record start time
start_time = time.time()

# Train with more OPPO+ algorithm concepts including multi-batch approach
total_timesteps = 300000 # setting the timesteps for model training
batch_size = model.n_steps # set the amount of steps required between each update
multi_batch_amnt = 2 # the batchs for the multi-batch approach
total_updates = total_timesteps//(batch_size*multi_batch_amnt) # round down for the calculation to get total updates
# perform the multi-batch updates based on above calculations
for update in range(total_updates):
    q_rewards = [] # reset the q reward values, or the values for expectation of future rewards
    augmented_rewards = [] # reset the reward values, or the actual rewards collected throughout the batch
    obs = env.reset() # reset the environment
    final_rewards = []
    # iterate through and collect data on an entire batch
    # for multi-batch approach
    for _ in range(multi_batch_amnt):
        batch_rewards = []
        done=False
        for _ in range(batch_size):
            # predict next action based on observation, with deterministic set to false.
            # If deterministic is set to true, then only selects action based on policy,
            # which encourages maximum exploitation. Here, deterministic = false, so the
            # model selects action based on some exploration/randomness, as desired by the
            # overall structure of OPPO+
            action, _states = model.predict(obs, deterministic=False) 
            obs, current_reward, done, info = env.step(action) # take action using the environment step
            # apply reward scaling and clipping based on the OPPO+ algorithm
            current_reward = np.clip(current_reward, -5, 5) / (np.std([current_reward]) + 1e-8)
            batch_rewards.append(current_reward)
            # calculate the new rewards and q reward values
            if (q_rewards): # if q_rewards, calculate new q reward
                new_q_reward = current_reward + (model.gamma*q_rewards[-1])
            else:
                new_q_reward = current_reward
            q_rewards.append(new_q_reward)
            # check completion status
            if done:
                obs = env.reset()
                break
        final_rewards.extend(batch_rewards)
    # use an OPPO+ bonus
    bonus_beta_value = 0.05 * (1-(update/total_updates))
    bonus_calculation = bonus_beta_value*np.std(final_rewards)
    augmented_rewards = [(i_reward+bonus_calculation+q_rewards[i]) for i, i_reward in enumerate(final_rewards)]
    # apply augmented rewards
    model.env.venv.envs[0].reward = np.mean(augmented_rewards)
    model.learn(total_timesteps=(batch_size*multi_batch_amnt), reset_num_timesteps=False)


# record end time
end_time = time.time()

# evaluate total training time
training_time = end_time - start_time

# save the model as .zip and environment statistics as .pkl for evaluation
model.save("outputs/phase4-model")
env.save("outputs/stats4.pkl")

# evalute model over different episode amounts
episode_amounts = [100, 1000, 5000]
with open("outputs/phase4.txt", "w") as FILEIN:
    FILEIN.write(f"Phase 4 (Combined Code-Level Optimizations and OPPO+ Algorithm) training completion time: {training_time:.3f} seconds\n")
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