# all neccessary imports for the environment setup for my
# minigrid robot-to-exit scene
import gymnasium as gym
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
import numpy as np
from gymnasium.spaces import Box
import random

"""
RobotToExitEnv class is for creation of the minigrid 2D environment
where the goal is to have the robot/agent navigate through the 2D
grid with obstacles and reach the goal/exit.
"""
class RobotToExitEnv(MiniGridEnv):

    def __init__(self, render_mode=None):
        # setting initial parameters for RobotToExit environment creation
        self.render_mode = render_mode # default to no visualization (do not display grid)
        self.size = 10 # sets the grid to 8x8 grid (actually is a 10x10 but walls cut it to 8x8)
        self.max_steps = 4*self.size*self.size # setting max_steps allotted to 4* grid space (4*10*10=400)
        self.seeThroughWalls = True # allows agent to see through/past walls for better observation of potential moves
        self.mission_space = MissionSpace(mission_func=lambda: "Agent move to green exit square.") # Set a mission space (goal of environment/test)

        # call parent MiniGridEnv library __init__ method to initalize environment
        super().__init__(
            render_mode = self.render_mode,
            grid_size = self.size,
            max_steps =  self.max_steps,
            see_through_walls = self.seeThroughWalls,
            mission_space = self.mission_space
        )
        
        # setting the agent's live observable space in the grid
        self.observation_space = Box(
            low=0,  # set agent's minimum visual observation color intensity to 0 (black)
            high=255, # set agent's maximum visual observation color intensity to 255 (white)
            shape=(7, 7, 3), # set agent's observable space to 7x7 grid space with 3 color-channel reading for the environment space (RGB)
            dtype=np.uint8 # use 8-bit unsigned integers for color values
        )

        # setting the value for average of grid cells that will contain obstacles
        self.obstacle_amnt = 0.025 # 2.5% average


    # MiniGridEnv._gen_grid has _gen_grid as an abstract method so we must implement it here.
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height) # set grid width and height
        self.agent_pos = (1, 1) # set agent start position
        self.agent_dir = 0 # set agent start direct (0=right facing)

        goal_pos = (width-2, height-2) # set exit/goal position accounting for walls
        self.put_obj(Goal(), *goal_pos) # put_obj creates an objective/goal in the environment
        self.grid.wall_rect(0, 0, width, height) # add wall boundary around the grid to restrict agent movement 


        # add walls to increase randomness/difficulty
        AGENT_ROW_MIN = 1
        AGENT_ROW_MAX = width-1
        AGENT_COL_MIN = 1
        AGENT_COL_MAX = height-1
        for i in range(AGENT_ROW_MIN, AGENT_ROW_MAX):
            for j in range(AGENT_COL_MIN, AGENT_COL_MAX):
                # ensure position != agent position or goal position
                if (i, j)!=self.agent_pos and (i, j)!=goal_pos:
                    # obstacle_amnt (2.5%) of times, generate a wall
                    if random.random()<self.obstacle_amnt:
                        self.put_obj(Wall(), i, j)

        self.mission = "Agent move to green exit square."

    # helper reset function for phases. calls the parent MiniGridEnv reset method
    # to re-initialize the environment
    def reset(self, **kwargs):
        # eset the environment and obtain data_obs (observation data) and
        # data_info (information dict of episode data like debug info) 
        data_obs, data_info = super().reset(**kwargs)
        # return only visual observation image data and the additional info dictionary
        visual_data_obs = data_obs['image']
        return visual_data_obs, data_info

    # helper step function for phases. calls the parent MiniGridEnv step method to
    # process the action, update the state, and return the result
    def step(self, action):
        # Execute the action and obtain data_obs (observation data),
        # data_reward_value (reward received), data_done (whether the episode has ended
        # as a result of reaching goal or failing), data_trunc (whether episode was
        # truncated due to max steps used), and data_info (additional info)
        data_obs, data_reward_value, data_done, data_trunc, data_info = super().step(action)
        data_visual_obs = data_obs['image']
        return data_visual_obs, data_reward_value, data_done, data_trunc, data_info
