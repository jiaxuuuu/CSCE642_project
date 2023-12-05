from scipy.optimize import minimize
import copy as py_copy
import math

def construct_obj(x, data_file, NUM_ACT, log=False):
    """
    Construct the objective function for optimization.

    Args:
    x (list): List of parameters to be optimized.
    data_file (str): Path to the data file containing observations.
    NUM_ACT (int): Number of actions.
    log (bool): Flag to enable logging.

    Returns:
    float: The objective value calculated from the data.
    """
    with open(data_file, 'r') as file:
        logl = 0  # Initialize log likelihood

        # Process each line in the file representing data for each timestep.
        for line_of_each_timestep in file:
            data = line_of_each_timestep.split()  # Split line into data points

            terms_for_all_instances = []

            # Process each instance in the data.
            for instance in data:
                info = instance.split(',')
                module_id = int(info[0])      # Module ID (e.g., 0)
                unit_reward = int(info[1])    # Unit reward (e.g., -1)
                chosen_action = int(info[2])  # Chosen action (e.g., 0)
                ds = [float(info[i+3]) for i in range(NUM_ACT)]  # Distances for each action

                # Calculate terms for each distance.
                # terms = [(x[module_id * 2] * unit_reward * (x[module_id * 2 + 1] ** d)) for d in ds]
                if module_id == 0:
                    terms = [(x[module_id * 2] * unit_reward * (x[module_id * 2 + 1] ** d))/5 for d in ds]
                elif module_id == 1:
                    terms = [(x[module_id * 2] * unit_reward * (x[module_id * 2 + 1] ** d)) for d in ds]

                terms_for_all_instances.append(py_copy.deepcopy(terms))

            # Calculate the first term of the log likelihood function.
            first_term = sum(terms[chosen_action] for terms in terms_for_all_instances)

            # Calculate the second term of the log likelihood function.
            second_term = math.log(sum(math.exp(sum(term[action_index] for term in terms_for_all_instances)) 
                                       for action_index in range(NUM_ACT)))

            # Update log likelihood.
            logl += (first_term - second_term)

    obj = -logl  # Objective is the negative log likelihood.

    # Optional logging.
    if log:
        print(obj)

    return obj

def optimize_for_two_modules(data_file, NUM_ACT, log):
    """
    Optimize the parameters for two modules.

    Args:
    data_file (str): Path to the data file.
    NUM_ACT (int): Number of actions.
    log (bool): Flag to enable logging.

    Returns:
    scipy.optimize.OptimizeResult: The result of the optimization.
    """
    # Initial guess and bounds for the parameters.
    x0 = [0.5, 0.5, 0.5, 0.5]
    bounds = [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99), (0.01, 0.99)]

    # Constraint ensuring the sum of weights for modules equals 1.
    cons = ({'type': 'eq', 'fun': lambda x: x[0] + x[2] - 1})

    # Perform the optimization.
    return minimize(construct_obj, x0, args=(data_file, NUM_ACT, log), method='SLSQP', bounds=bounds, constraints=cons)
