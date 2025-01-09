from gymnasium.envs.registration import register

register(
    id='MiniGrid-RobotToExit-v0', # environment identifier, used with gym.make('MiniGrid-RobotToExit-v0') comands 
    entry_point='envs.robot_to_exit_env:RobotToExitEnv', # path to environment class
)