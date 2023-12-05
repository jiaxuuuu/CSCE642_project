import numpy as np
from tqdm import tqdm
import gymnasium as gym
import interaction
from ActionSelection import actionSelection
import time

obstacle_poses = [ (4.69, -5.86),(4.79, -9.78),(10.48, -12.59),(14.05, -11.09),(10.15, -9.40), (12.94, -6.36),(8.04, -5.28)]
""" Step 1: Makes an environment instance """
# Please apply 'final_gym_env_id' to here.
env = gym.make("ConstructionSite-v1_seed_None")

# Initialize trajectories list
trajectories = []

# Starts Socket Communication with Unity
""" Step 2: Press the play button on Unity and execute this section. """
# Until you get the message "Successfully connect to Unity!"
connection_result = env.unwrapped.connect_and_check_unity()
print(connection_result)
if not connection_result:
    print("It seems Unity has been disconnected. Gym env will be closed.")
    env.close()

""" Step 3: Define the total number of episodes (iteration)"""
# Perform a single episode with user's keyboard input
num_episodes = 30

for episode in tqdm(range(num_episodes)):
    obs, info = env.reset()  # Reset the environment and retrieve the initial observation.
    print(info)
    done = False


    # Execute one episode until termination or truncation.
    while not done:

        stepsize = 0.3
        # agressive
        # r1 = (0.98+0.66+0.99)/3
        # r2 = (0.34+0.02+0.01)/3
        # gamma_1 = (0.78+0.77+0.76)/3
        # gamma_2 = (0.67+0.79+0.42)/3


        # conservative
        r1 = (0.01+0.86+0.78)/3
        r2 = (0.22+0.14+0.99)/3
        gamma_1 = (0.75*2+0.77)/3
        gamma_2 = (0.79+0.84+0.86)/3

        # r1 = 0.01
        # r2 = 0.99
        # gamma_1 = 0.75
        # gamma_2 = 0.86

        # r1 = 0.70
        # r2 = 0.30
        # gamma_1 = 0.75
        # gamma_2 = 0.79

        """ Step 4-2: action should be provided from the policy."""
        """ You can provide an action by defining methods from 'interaction.py """
        action = actionSelection(obs, stepsize, obstacle_poses, r1,gamma_1, r2, gamma_2)

        # Get action from user's keyboard input
        # print("Now you can start press action")
        # action = env.unwrapped.get_action_from_user() # heuristic
        if action!=8:
            print(action)
            next_obs, reward, terminated, truncated, info = env.step(action)
    #
            # Store the result in trajectories list
            trajectories.append((obs, action, reward))

            obs = next_obs

            print(obs[0])

            # Check if the episode has ended and update the current observation.
            if (terminated or truncated):
                done = 1
                env.send_screenshot_message()
                time.sleep(3)
                env.send_done_message()
                time.sleep(3)
#
# Notify Unity that the training is complete, prompting it to exit play mode or shut down the built application.
env.send_message_training_over()
#
# Also close the environment defined in Python.
env.close()