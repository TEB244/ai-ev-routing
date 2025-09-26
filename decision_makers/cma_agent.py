# Crated by Santiago July 19, 2024
# Updated August 9, 2024
import cma
import numpy as np
import torch
import pickle
import copy

from environment.data_loader import load_config_file
INF_THRESHOLD = 1e+07

class CMAAgent:
    """
    A class to represent a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) agent
    for optimizing decision-making models in electric vehicle routing and charging scenarios.

    Attributes:
        population_size (int): Size of the population used in the evolution strategy.
        max_generation (int): Maximum number of generations for the evolution process.
        es (cma.CMAEvolutionStrategy): Instance of the CMA-ES algorithm.
        weights (list): List to store weights during optimization.
        in_size (int): Dimension of the state input.
        out_size (int): Dimension of the action output.
        num_cars (int): Number of cars involved in the simulation.
        cma_config (dict): Configuration dictionary for the CMA-ES.
        initial_sigma (float): Initial step size for the CMA-ES algorithm.
        model_type (str): Type of model being optimized (e.g., 'optimizer', 'NN_basic').
        states (list): List to store states during optimization.
        actions (list): List to store actions taken during optimization.
        weights_result (list): List to store the resulting weights after optimization.
    """

    def __init__(self, state_dimension, action_dimension, num_cars, seed, agent_index, global_weights, experiment_number, device, dtype):
        """
        Initializes the CMAAgent with configuration and parameters for the CMA-ES algorithm.

        Parameters:
            state_dimension (int): The dimensionality of the input state.
            action_dimension (int): The dimensionality of the output action.
            num_cars (int): The number of cars involved in the scenario.
            seed (int): Seed for random number generation to ensure reproducibility.
            agent_index (int): Index of the agent in a multi-agent scenario.
            global_weights (dict): Pre-trained global weights for initializing the agent's model.
        """
        
        # Load CMA-ES configuration parameters from a YAML file
        fname = f'experiments/Exp_{experiment_number}/config.yaml'
        c = load_config_file(fname)
        cma_c = c['cma_parameters']

        # Set up CMA-ES parameters
        self.population_size = cma_c['population_dimension']
        self.max_generation = cma_c['max_generations']
        initial_sigma = cma_c['initial_sigma'] 
        model_type = cma_c['model_type']

        # Initialize random number generator with a seed
        rng = np.random.default_rng(seed)

        # Select and initialize the model type to be optimized with CMA-ES
        if model_type == 'optimizer':
            # Optimizer model does not use state information, directly optimizes weights
            initial_weights = rng.random(action_dimension)
            bounds = [0, 1]
            # bounds = [0, INF_THRESHOLD]
            self.model = self.cma_model

        elif model_type == 'NN_basic':
            # Simple neural network model (initialization to be implemented)
            initial_weights = np.array(rng.random(state_dimension) * 2 - 1)

        else:
            raise RuntimeError('CMA model type not selected or incorrect model in CMA config file.')

        # If global weights are provided, use them to initialize the agent's weights
        if global_weights is not None:
            initial_weights = torch.tensor([tensor for key, tensor in global_weights.items()]).tolist()

        # Initialize the CMA-ES algorithm with the configuration and initial weights
        cma_config = {
            'popsize': self.population_size,
            'maxiter': self.max_generation,
            'bounds': bounds,
            'seed': seed,
            'verb_disp': 0
        }
        es = cma.CMAEvolutionStrategy(initial_weights, initial_sigma, cma_config)

        # Store relevant parameters and objects for the agent
        self.device = device
        self.dtype = dtype
        self.es = es
        self.weights = []
        self.in_size = state_dimension
        self.out_size = action_dimension
        self.num_cars = num_cars
        self.cma_config = cma_config
        self.initial_sigma = initial_sigma
        self.model_type = model_type
        self.states = []
        self.actions = []
        self.gen = 0
        self.saved_solutions = []
        self.restore_solution =  False
        self.weights_result = torch.zeros((self.max_generation,len(initial_weights)),\
                                          device=device, dtype=dtype)

    def cma_model(self, state, weights):
        """
        A simple CMA-ES optimizer model that directly optimizes weights without using the state input.

        Parameters:
            state (ndarray): The current state (not used in this model).
            weights (ndarray): The weights to be optimized.

        Returns:
            ndarray: The optimized weights.
        """
        solutions = weights
        return solutions

    def cma_restart(self):
        """ Restart the cma evolution strategies model because of reaching the matrix max limit
        """
        last_weights = self.es.best.x
        self.es = None
        es = cma.CMAEvolutionStrategy(last_weights, self.initial_sigma, self.cma_config)
        self.es = es
        last_weights = None
    
    def get_solutions(self):
        """
        Requests a set of candidate solutions (weights) from the CMA-ES algorithm.

        Returns:
            ndarray: A set of candidate solutions for the current generation.
        """
        solutions = torch.tensor(self.es.ask(), device=self.device, dtype=self.dtype)
        # # print(f'solutions shape {solutions.shape}')
        # # solutions[1,1] = INF_THRESHOLD+1
        # # Mask for values exceeding threshold
        # self.mask = solutions > INF_THRESHOLD
        # # self.restore_solution = False

        # if self.mask.any():
        #     self.map_solutions = torch.zeros((solutions.shape), dtype=self.dtype)
        #     self.map_solutions[self.mask] = solutions[self.mask]
        #     solutions[self.mask] = float('inf')
        #     print(f'modifying inf solutions with mask {self.mask} and\n saving values {self.map_solutions}')
        #     # else:
        # #     self.saved_solutions = solutions

        #     # # Store original values and positions for restoration later
        #     # for i in range(solutions.shape[0]):
        #     #     # if mask[i].any():
        #     #     #     print(f'mask i {mask[i]}')
        #     #     #     self.restore_map[i] = solutions[i].clone()
        #     #     #     print(f'restore map {self.restore_map}')
        #     #     #     solutions[i][mask[i]] = float('inf')
        #     #     print()
        return solutions

    def get_best_solutions(self):
        """
        Retrieves the best solution (weights) found by the CMA-ES algorithm so far.

        Returns:
            ndarray: The best weights found during optimization.
        """
        weights = torch.from_numpy(self.es.best.x).to(device=self.device)
        self.weights_result[self.gen] = weights
        self.gen =+ 1
        return weights

    def get_weights(self):
        """
        Converts the best solution's weights into a dictionary format for easier retrieval.

        Returns:
            dict: A dictionary where keys are weight names and values are the corresponding weights.
        """
        weights = []
        weights_last_step = self.weights_result[self.gen-1,:]
        keys = [f'Weight {w}' for w in range(len(weights_last_step))]
        weights_step = dict(zip(keys, torch.tensor(weights_last_step)))

        return weights_step

    def tell(self, solutions, reward):
        """
        Informs the CMA-ES algorithm about the fitness (reward) of the current generation's solutions.

        Parameters:
            reward (ndarray): The fitness values associated with the solutions of the current generation.
        """
        # if self.mask.any():
        #     # solutions[self.mask] = self.map_solutions[self.mask]
        #     solutions[self.mask] = INF_THRESHOLD

        self.es.tell(solutions.cpu().numpy(), reward.cpu().numpy())


    def save_model(self, fname):
        """
        Saves the current state of the CMA-ES optimizer to a file.

        Parameters:
            fname (str): The file name where the optimizer's state will be saved.
        """
        with open(fname, 'wb') as f:
            pickle.dump(self.es, f)