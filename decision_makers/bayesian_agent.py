import torch
from skopt.space import Real

def create_search_space(action_dim, range_min, range_max):
    """
    Creates the search space for the Bayesian Optimizer.

    Parameters:
        action_dim (int): The number of dimensions (weights) to optimize.
        range_min (float): The minimum value for each weight.
        range_max (float): The maximum value for each weight.

    Returns:
        list: A list of skopt.space.Real dimensions.
    """
    search_space = []
    for i in range(action_dim):
        search_space.append(Real(range_min, range_max, name=f'weight_{i}'))
    return search_space

def get_best_action_from_result(result, device, dtype):
    """
    Extracts the best action (weight vector) from the optimizer's result.

    Parameters:
        result (scipy.optimize.OptimizeResult): The result object from gp_minimize.
        device (str): The device to place the resulting tensor on.
        dtype (torch.dtype): The data type for the resulting tensor.

    Returns:
        torch.Tensor: The best weight vector found by the optimizer.
    """
    # result.x contains the list of best parameter values found
    best_weights = torch.tensor(result.x, dtype=dtype, device=device)
    return best_weights