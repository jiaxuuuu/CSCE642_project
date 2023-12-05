import numpy as np
import pandas as pd

def multiply_location_change_per_axis(row, walking_stride):
    """
    Modify 'Agent Location Change' by multiplying each axis by 'walking_stride'.

    Args:
    row (pd.Series): A row from a DataFrame.
    walking_stride (float): Multiplier for the walking stride.

    Returns:
    str: Modified 'Agent Location Change' as a string.
    """
    axes = row['Agent Location Change'].split()
    multiplied_axes = [str(round(float(axis) * walking_stride, 6)) for axis in axes]
    return ' '.join(multiplied_axes)

def calculate_state_difference(df, mode):
    """
    Calculate the state difference per axis.

    Args:
    df (pd.DataFrame): DataFrame to perform calculations on.
    mode (str): Operation mode - 'fromActionToNaked' or 'fromNakedToAction'.

    Returns:
    pd.DataFrame: Modified DataFrame with updated positions.
    """
    for axis in ['x', 'y', 'z']:
        pos_axis = 'pos_' + axis 
        diff_axis = 'diff_' + axis

        if mode == 'fromActionToNaked':
            df[pos_axis] = df[pos_axis] - df[diff_axis]
        elif mode == 'fromNakedToAction':
            df[pos_axis] = df[pos_axis] + df[diff_axis]
    
    return df

def update_eu_distance_2d(df):
    """
    Update the Euclidean distances for each object based on the agent's position.

    Args:
    df (pd.DataFrame): DataFrame with agent positions.
    object_positions (dict): Dictionary of object positions.

    Returns:
    pd.DataFrame: DataFrame with updated distances.
    """

    # Fixed objects' location
    pos_target = np.array([4.69,0,-5.86])
    pos_building = np.array([4.79,0,-9.78])
    pos_ob1 = np.array([10.48,0,-12.59])
    pos_ob2 = np.array([14.05,0,-11.09])
    pos_ob3 = np.array([10.15,0,-9.4])
    pos_ob4 = np.array([12.94,0,-6.36])
    pos_ob5 = np.array([8.04,0,-5.28])

    # Function to calculate Euclidean distance
    def calculate_eu_distance_2d(row, vector_obj):
        """
        Calculate Euclidean distance in 2D (x and z axes).

        Args:
        row (pd.Series): A row from a DataFrame containing 'pos_x' and 'pos_z'.
        vector_obj (np.array): A numpy array representing the target object's position in 2D.

        Returns:
        float: The Euclidean distance.
        """
        vector_agnt = np.array([row['pos_x'] , row['pos_z']]) # 2D
        vector_obj_xz = np.array([vector_obj[0], vector_obj[2]]) # 2D
        return np.linalg.norm(vector_agnt - vector_obj_xz)

    # Apply the function to each row and update 'd2obj'
    df['d2target'] = df.apply(calculate_eu_distance_2d, vector_obj=pos_target, axis=1)
    df[' d2building'] = df.apply(calculate_eu_distance_2d, vector_obj=pos_building, axis=1)
    df[' d2obstacle1'] = df.apply(calculate_eu_distance_2d, vector_obj=pos_ob1, axis=1)
    df[' d2obstacle2'] = df.apply(calculate_eu_distance_2d, vector_obj=pos_ob2, axis=1)
    df[' d2obstacle3'] = df.apply(calculate_eu_distance_2d, vector_obj=pos_ob3, axis=1)
    df[' d2obstacle4'] = df.apply(calculate_eu_distance_2d, vector_obj=pos_ob4, axis=1)
    df[' d2obstacle5'] = df.apply(calculate_eu_distance_2d, vector_obj=pos_ob5, axis=1)

    return df

def generate_dfs_for_all_actions(df_naked, dB, actions):
    """
    Generate dataframes for all actions.

    Args:
    df_naked (pd.DataFrame): Base DataFrame without actions.
    dB (pd.DataFrame): DataFrame with agent location changes.
    actions (list): List of possible actions.

    Returns:
    list: List of DataFrames for each action.
    """

    # A list of dataframes, each representing a specific action.
    # For example, dfs[0] refers to records when take action '0' at all timesteps.
    dfs = []
    for action in actions: # action = 0, 1, 2, ..., 7
        df_action = df_naked.copy() # new dataframe the same as the naked df
        df_action['Action'] = action # change actions in all rows to the 'action'

        # create columns diff_x, diff_y, diff_z of 'action'.
        dB_split_xyz = dB['Agent Location Change'][action].split()
        df_action[['diff_x', 'diff_y', 'diff_z']] = [float(val) for val in dB_split_xyz]

        # pos_x,y,z = (pos_x,y,z + diff_x,y,z) <-- Now pos_x,y,z refers to states after taking 'action'.
        # Calculate state difference per axis
        df_action = calculate_state_difference(df_action, 'fromNakedToAction')

        # update distance by using updated agent position.
        df_action = update_eu_distance_2d(df_action)

        dfs.append(df_action)

    # Now, dfs is df list in which each df refers to distances when a certain action has been chosen and agent state after taking the action.
    # make these codes optimized via GPT with defining functions.

    return dfs

def create_text_line(row, dfs, actions, unitRwdM1, unitRwdM2):
    """
    Create a text line containing module information, unit rewards, action values, and distances.

    Args:
    row (pd.Series): A row from a DataFrame, representing a single data instance.
    dfs (list of pd.DataFrame): List of dataframes, each corresponding to a specific action.
    actions (list): List of possible actions.

    Returns:
    str: A string representation of the combined information for each module instance.
    """

    # Initialize a list to hold chunks of text for each module instance.
    chunks = []

    # Get the index of the current row in the DataFrame.
    current_index = row.name

    instance_num = 7

    # Loop over each module instance.
    for i in range(instance_num): # There are 'instance_num' instances in total.
        if (i==0):
            module_id = 0
            unit_reward = unitRwdM1 # For taking the shortest path.
        elif (i==1):
            module_id = 1
            unit_reward = unitRwdM2 # For avoiding obstacles from here.
        elif (i==2):
            module_id = 1
            unit_reward = unitRwdM2
        elif (i==3):
            module_id = 1
            unit_reward = unitRwdM2
        elif (i==4):
            module_id = 1
            unit_reward = unitRwdM2
        elif (i==5):
            module_id = 1
            unit_reward = unitRwdM2
        elif (i==6):
            module_id = 1
            unit_reward = unitRwdM2
        
        # Initialize a chunk with module ID, unit reward, and the action taken.
        chunk = [str(module_id), str(unit_reward), str(int(row['Action']))]
        
        # Initialize a list to hold random values for each action.
        random_values = [str(1) for _ in range(len(actions))]

        # Loop over each action to determine the value to append to the chunk.
        for action in actions:
            if action == int(row['Action']):
                # If the current action is the one taken, use the actual distance value from 'row'.
                if i == 0:  
                    random_values[action] = str(round((row['d2target']), 6))
                elif i == 1:  
                    random_values[action] = str(round((row[' d2building']), 6))
                elif i == 2:  
                    random_values[action] = str(round((row[' d2obstacle1']), 6))
                elif i == 3:  
                    random_values[action] = str(round((row[' d2obstacle2']), 6))
                elif i == 4:  
                    random_values[action] = str(round((row[' d2obstacle3']), 6))
                elif i == 5:  
                    random_values[action] = str(round((row[' d2obstacle4']), 6))
                elif i == 6:
                    random_values[action] = str(round((row[' d2obstacle5']), 6))
            else: 
                # If the action is not the one taken, use the distance value from the corresponding DataFrame in 'dfs'.
                if i == 0:
                    random_values[action] = str(round(dfs[action].loc[current_index, "d2target"], 6))
                elif i == 1:
                    random_values[action] = str(round(dfs[action].loc[current_index, " d2building"], 6))
                elif i == 2:
                    random_values[action] = str(round(dfs[action].loc[current_index, " d2obstacle1"], 6))
                elif i == 3:
                    random_values[action] = str(round(dfs[action].loc[current_index, " d2obstacle2"], 6))
                elif i == 4:
                    random_values[action] = str(round(dfs[action].loc[current_index, " d2obstacle3"], 6))
                elif i == 5:
                    random_values[action] = str(round(dfs[action].loc[current_index, " d2obstacle4"], 6))
                elif i == 6:
                    random_values[action] = str(round(dfs[action].loc[current_index, " d2obstacle5"], 6))

        # Extend the chunk with the calculated random values and append to chunks list.
        chunk.extend(random_values)
        chunks.append(','.join(chunk))
    
    # Combine all chunks into a single string.
    return ' '.join(chunks)