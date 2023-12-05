"""
This module serves to:
1) Define the state space, action space, and reward functions.
2) Register the pre-configured Unity environment into the Gym environment.
3) Facilitate interactions between the agent and the Unity environment.

Written by Minguk Kim @ Texas A&M University
mingukkim@tamu.edu
"""

from __future__ import annotations
import keyboard
import time
import numpy as np

# Importing required libraries for reinforcement learning
import gymnasium as gym
from gymnasium import spaces

# Importing library for Python-Unity communication
from communication import UnityCommunication

# Registering the pre-defined Unity environment with the local Gym
from gymnasium.envs.registration import register

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
BELOW ARE VARIABLE(S) FOR REGISTERING UNITY ENVIRONMENT TO GYM
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
gym_env_id = 'ConstructionSite-' # STEP 1
gym_env_version = 'v1' # STEP 2
max_episode_steps = 1_000 # STEP 3 -  how many episode do we going to continue?
max_timesteps_per_episode = 2000 # STEP 4 - how many timesteps does a single episode have?
global_seed = None # STEP 5 - If none, random seed.
final_gym_env_id = gym_env_id + gym_env_version + '_seed_' + str(global_seed)
print("final_gym_env_id: {}".format(final_gym_env_id))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
DON'T NEED TO TOUCH THE CODES BELOW.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
register(
    id=final_gym_env_id,
    entry_point='interaction:UnityEnv', # 'filename:classname'
    max_episode_steps= max_episode_steps,
    kwargs={'reward_matrix': None}
)

class UnityEnv(gym.Env):
    """
    A custom environment class that interfaces with a Unity environment for reinforcement learning.

    Attributes:
    - reward_matrix (array-like): Matrix that denotes rewards for different states.
    ...
    """
    
    def __init__(self, reward_matrix=None):
        super(UnityEnv, self).__init__()

        """""""""""""""""""""""""""""""""""""""
        Defines state and action spaces
        """""""""""""""""""""""""""""""""""""""
        # STEP 6 – define state space type and size.
        # a continuous space of 32 dimensions
        state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32) 

        # STEP 7 – define action space type and size.
        action_space = spaces.Discrete(8)
        
        """""""""""""""""""""""""""""""""""""""
        Defines initial variables for RL
        """""""""""""""""""""""""""""""""""""""
        self.episode_rewards = []
        self.reward_matrix = reward_matrix # Dont worry about this. We are not gonna use this.

        """""""""""""""""""""""""""""""""""""""
        DON'T NEED TO TOUCH THE CODES BELOW.
        """""""""""""""""""""""""""""""""""""""
        self.unity_comm = UnityCommunication() # Initialize communication with Unity
        self.observation_space = state_space
        self.action_space = action_space
        self.isRecording = False
        self.current_timestep = 0
        self.np_random = None
        self.seed = self.seed(global_seed)
        self.max_timesteps = max_timesteps_per_episode
        
    def reset(self, **kwargs):
        """
        Resets the environment to its initial state.
        To initializing env and to start each episode, this method will be executed.
        This method returns observation results of initial states of the env.

        # STEP 8 - You don't need to define initial state here. 
        # Just make sure that 'max_timesteps_per_episode' here 
        # and your Unity script has the same number.
        
        Returns:
        - observation (array-like): The starting state of the environment.
        - {} (dict): Empty dictionary for API consistency.
        """

        # reset step size
        self.current_timestep = 0
        # seed_value = kwargs.get('seed')
        seed_value = global_seed
        if seed_value is not None:
            self.seed(seed_value)
        
        # agent's location at random at every initial timestep.
        # for example,
        # self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        
        observation = self._get_obs()
        #print("observation: {}".format(observation))
        info = self._get_info()
        
        self.current_timestep += 1
        
        return observation, info
    
    def step(self, action):
        """
        Update the agent's state based on the given action by communicating with Unity.
        
        Parameters:
        - action (int): Action chosen by the agent.
        
        Returns:
        - observation (array-like): The new state after the action.
        - reward (float): Immediate reward after the action.
        - terminated (bool): Whether the episode has ended.
        - truncated (bool): Whether the episode was truncated before reaching the end.
        - {} (dict): Empty dictionary for API consistency.
        """
        
        """
        STEP 9 - Please redefine these actions.
        You have to redefine the script 'RLAgentCommunicator.cs' together.
        """
        try:
            # Update agent position based on action and send it to Unity via TCP
            if action == 0:  # North
                self.unity_comm.send_action_to_unity(0)
            elif action == 1:  # Northeast
                self.unity_comm.send_action_to_unity(1)
            elif action == 2:  # East
                self.unity_comm.send_action_to_unity(2)
            elif action == 3:  # southeast
                self.unity_comm.send_action_to_unity(3)
            elif action == 4:  # south
                self.unity_comm.send_action_to_unity(4)
            elif action == 5:  # southwest
                self.unity_comm.send_action_to_unity(5)
            elif action == 6:  # west
                self.unity_comm.send_action_to_unity(6)
            elif action == 7:  # northwest
                self.unity_comm.send_action_to_unity(7)
            # elif action == 8:  # Do Nothing
            #     self.unity_comm.send_action_to_unity(8)

            observation = self._get_obs()


            # Reward calculation and episode termination conditions
            if self.reward_matrix == None:
                """
                STEP 10 - Define rewards
                You may refer to the example codes below.

                Also, please modify the condition of 'terminiated' if necessary.
                """

                # example codes of reward function.
                # define rewards
                reward = 100.0 if (observation[2] < 1.2) else -1
                
                # reward logs
                # observation[2] is the distance to the target
                if (observation[2] < 1.2):
                    print("Success! Reward +100")

            else:
                pass
            
            self.episode_rewards.append(reward)
            
            self.current_timestep += 1

            # print("observation[0]: {}".format(observation[0]))
            # observation[0] (hasDumped) = 0 means False. i.e., the rock not yet dumped.
            # a condition for early epsiode termination.
            terminated = True if (observation[2] < 1.2) else False
            # default termniation condition.
            truncated = self.current_timestep >= self.max_timesteps

            return observation, reward, terminated, truncated, {}
        except BrokenPipeError:
            print("Connection lost while trying to send data: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def set_reward_matrix(self, reward_matrix):
        """
        Update the reward matrix for the environment.
        
        Parameters:
        - reward_matrix (array-like): The new reward matrix to be set.
        """
        self.reward_matrix = reward_matrix
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    BELOW ARE METHOD(S) FOR HEURISTIC ACTIONS
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def get_action_from_user(self):
        """
        STEP 11 - This is for controling agent manually. Please modify this also.

        Fetches an action based on user keyboard input.

        Returns:
        - action (int): The action corresponding to the key pressed by the user.

        0: North (W)
        1: Northeast (E)
        2: East (D)
        3: Southeast (X)
        4: South (S)
        5: Southwest (Z)
        6: West (A)
        7: Northwest (Q)
        """
        sleep_time = 0.05
        # sleep_time = 0.1
        time.sleep(sleep_time)  # Provide a slight delay to detect key presses effectively
        try:
            while True:
                if keyboard.is_pressed('W'):
                    print("Key W pressed!")
                    time.sleep(sleep_time)
                    return 0
                elif keyboard.is_pressed('E'):
                    time.sleep(sleep_time)
                    return 1
                elif keyboard.is_pressed('D'):
                    time.sleep(sleep_time)
                    return 2
                elif keyboard.is_pressed('X'):
                    time.sleep(sleep_time)
                    return 3
                elif keyboard.is_pressed('S'):
                    time.sleep(sleep_time)
                    return 4
                elif keyboard.is_pressed('Z'):
                    time.sleep(sleep_time)
                    return 5
                elif keyboard.is_pressed('A'):
                    time.sleep(sleep_time)
                    return 6
                elif keyboard.is_pressed('Q'):
                    time.sleep(sleep_time)
                    return 7
                elif not any(keyboard.is_pressed(key) for key in ['W', 'E', 'D', 'X', 'S', 'Z', 'A', 'Q']):
                    time.sleep(sleep_time)
                    return 8
        except AttributeError as e:
            print(f"get action from user method has an error: {e}")

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    BELOW ARE METHOD(S) FOR COMMUNICATION
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def connect_and_check_unity(self):
        """Establish a connection to Unity and validate it."""
        return self.unity_comm.connect_and_check_unity()

    def send_message_training_over(self):
        """Send a notification to Unity to inform that training has completed."""
        self.unity_comm.send_message_training_over()

    def send_done_message(self):
        """Send done message to unity and then unity restart a episode"""
        self.unity_comm.send_done_message()

    def send_screenshot_message(self):
        self.unity_comm.send_screenshot_message()
        print("Take picture.")



    def seed(self, seed=None):
        """
        Set the seed for this environment's random number generator.
        
        Parameters:
        - seed (int): The seed value.
        
        Returns:
        - [seed] (list of int): List containing the seed value.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    DEFINES STATES THAT ARE NEED TO BE RETRIEVED.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def _get_obs(self):
        """
        This method retunrs observed states fron Unity.
        Since we will need to compute observations both in reset and step, 
        it is often convenient to have a (private) method _get_obs that 
        translates the environment’s state into an observation.
        """

        var_dic = {
            "currentPosition":"xz",
            "d2target": "dis",
            "d2building": "dis",
            "d2obstacle1": "dis",
            "d2obstacle2": "dis",
            "d2obstacle3": "dis",
            "d2obstacle4": "dis",
            "d2obstacle5": "dis"
        }
        
        # Receives object info as state from Unity by traversing the dictionary
        observation_list = []
        for v_name, vType in var_dic.items():
            state = self.unity_comm.receive_state_info(v_name, vType)
            print("The state of {} is {}".format(v_name,state))
            observation_list.extend(state)
        
        # Stores observed data
        observation = np.array(observation_list, dtype=np.float32) # observe x,y,z
        
        # Debugging: Print the length and a sample of the observation array
        # print(f"Observation length: {len(observation)}")
        # print(f"Observation sample: {observation}")
        
        # returns observation
        return observation
        
    
    def _get_info(self):
        """
        Oftentimes, info will also contian some data that is only available insdie the 'step' method.
        
        Returns:
        - returned value should be dictionary type
        """
        
        # return {
        # "distance": np.linalg.norm(
        #     self._agent_location - self._target_location, ord=1
        # )
        # }
        return {}