import math
import numpy as np



def distance_next(pose, action, stepsize, obstacle_poses):
    # Define the action effects
    action_effects = {
        0: (0, 1),       # North
        1: (math.sqrt(0.5), math.sqrt(0.5)), # Northeast
        2: (1, 0),       # East
        3: (math.sqrt(0.5), -math.sqrt(0.5)),# Southeast
        4: (0, -1),      # South
        5: (-math.sqrt(0.5), -math.sqrt(0.5)),# Southwest
        6: (-1, 0),      # West
        7: (-math.sqrt(0.5), math.sqrt(0.5)) # Northwest
    }

    # Calculate the new position after taking the action
    dx, dz = action_effects[action]
    new_position = (pose[0] + dx * stepsize, pose[1] + dz * stepsize)

    # Function to calculate Euclidean distance between two points
    def euclidean_distance(pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    # Calculate the distances from the new position to each obstacle
    new_distances = [euclidean_distance(new_position, obstacle) for obstacle in obstacle_poses]

    return new_distances


def Q_cal(reward, gamma, dis):
    # Check if dis is a single number or a list
    if isinstance(dis, (int, float)):  # dis is a single number
        Q = reward * (gamma ** dis)
    elif isinstance(dis, list):  # dis is a list
        Q = sum(reward * (gamma ** (d)) for d in dis)
    else:
        raise ValueError("Invalid type for dis. Must be a number or a list of numbers.")

    return Q


def softmax_action_selection(Q_values):
    # Apply softmax to convert Q-values to probabilities
    probabilities = np.exp(Q_values) / np.sum(np.exp(Q_values))

    # Select an action based on these probabilities
    action = np.random.choice(len(Q_values), p=probabilities)

    return action

def greedy_selection(Q_values):
    # Find the action with the highest Q-value
    action = np.argmax(Q_values)
    return action




def actionSelection(state, stepsize, obstacle_poses, r1,gamma_1, r2, gamma_2):
    # state = [pose_x,pose_z, d2target, d2otherobstacles]
    # calculate next dis for each action
    Q_values=[]
    for action in range (8):
        pose = (state[0], state[1])
        dis_next = distance_next(pose, action, stepsize, obstacle_poses)
        Q1 = Q_cal(r1, gamma_1, dis_next[0])/5
        # Q2 = -1*Q_cal(r2, gamma_2, (min(dis_next[1:])-3))
        Q2 = -Q_cal(r2, gamma_2, dis_next[1:])
        print("Q1: {}, Q2: {}".format(Q1, Q2))
        Q = Q1+Q2
        print()
        Q_values.append(Q)
    action = softmax_action_selection(Q_values)
    # action = greedy_selection(Q_values)
    return action
