
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    """
    A neural network for the REINFORCE algorithm that outputs action probabilities.
    """
    def __init__(self, state_dim, action_dim, layers):
        super(PolicyNetwork, self).__init__()

        self.layers = nn.ModuleList()
        for i, layer_size in enumerate(layers):
            if i == 0:
                linear_layer = nn.Linear(state_dim, layer_size)
            else:
                linear_layer = nn.Linear(layers[i - 1], layer_size)
            
            self.layers.append(linear_layer)

        self.output = nn.Linear(layers[-1], action_dim)
        
    def forward(self, state):
        x = state
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = torch.relu(x)  # Apply ReLU activation
        x = self.output(x)
        probs = F.softmax(x, dim=-1)
        return probs

def initialize(state_dim, action_dim, layers, device_agents):
    """
    Initializes the PolicyNetwork for the REINFORCE agent.

    Parameters:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        layers (list): List of integers defining the architecture of the neural networks.
        device_agents (torch.device): The device to which the policy network will be moved.

    Returns:
        policy_network (PolicyNetwork): Initialized PolicyNetwork.
    """
    policy_network = PolicyNetwork(state_dim, action_dim, layers)
    return policy_network.to(device_agents)

def compute_loss(experiences, gamma, policy_network):
    """
    REINFORCE loss with two standard variance-reduction tweaks:

    1. **Baseline subtraction**: subtract the batch mean from the discounted
       returns so the gradient is driven by the *advantage* (how much better
       than typical this action was) rather than absolute return magnitude.
       Without this, the per-episode rewards in [-700, -40] dominate the
       gradient regardless of which action was selected, masking the
       per-action learning signal.

    2. **Numerical safety on log**: clamp the chosen probability so that a
       saturated softmax (prob -> 0) does not produce log(0) = -inf and
       poison the gradient.

    Parameters:
        experiences (tuple):
            - states  (torch.tensor): batch of states.
            - actions (torch.tensor): batch of stored action probability
                                       vectors (output of the policy at
                                       action-collection time).
            - rewards (torch.tensor): batch of rewards.
            - dones   (torch.tensor): batch of done flags.
        gamma (float): discount factor.
        policy_network (PolicyNetwork): policy being trained.

    Returns:
        torch.tensor: scalar policy-gradient loss.
    """

    states, actions, rewards, dones = experiences

    # Discounted returns (Monte Carlo).
    returns = []
    G = 0
    for r, done in zip(reversed(rewards), reversed(dones)):
        G = r + gamma * G * (1 - done)
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32, device=states.device)

    # Baseline subtraction: use the batch mean as a state-independent baseline.
    # This is the simplest control variate; it does not change the gradient's
    # expected value but reduces its variance substantially when rewards are
    # large and similar across samples.
    if returns.numel() > 1:
        returns = returns - returns.mean()

    # Current policy's probability distribution over actions.
    probs = policy_network(states)

    # Action chosen at collection time: argmax over the stored output vector.
    action_indices = torch.argmax(actions, dim=-1)

    # Probability of the chosen action, clamped for numerical safety.
    chosen_probs = torch.gather(probs, 1, action_indices.unsqueeze(1)).squeeze(1)
    chosen_probs = chosen_probs.clamp(min=1e-8)
    log_probs = torch.log(chosen_probs)

    # REINFORCE: -E[log pi(a|s) * advantage].
    loss = -(log_probs * returns).mean()
    return loss

def agent_learn(experiences, gamma, policy_network, optimizer, device):
    """
    Performs a learning step for the REINFORCE agent by computing the loss and updating the policy network's weights.

    Parameters:
        experiences (tuple): A tuple containing:
            - states (numpy.array): Batch of states.
            - actions (numpy.array): Batch of actions taken.
            - rewards (numpy.array): Batch of rewards.
            - next_states (numpy.array): Batch of next states (not used in REINFORCE).
            - dones (numpy.array): Batch of done flags indicating episode termination.
        gamma (float): Discount factor for future rewards.
        policy_network (PolicyNetwork): The policy network to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating the policy network's weights.
        device (torch.device): The device to which the tensors will be moved.

    Returns:
        None
    """

    loss = compute_loss(experiences, gamma, policy_network)

    optimizer.zero_grad()
    loss.backward()
    # Apply gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=1.0)
    optimizer.step()

def get_actions(state, policy_networks, episode_index, agent_index, device, epsilon, random_threshold, nn_by_zone):
    """
    Selects actions for an agent using a mixture of distribution sampling and greedy approach
    (based on an epsilon threshold).

    Parameters:
        state (torch.tensor): The current state of the agent.
        policy_networks (list): List of policy networks for each agent.
        episode_index (int): Index of the current episode.
        agent_index (int): Index of the current agent.
        device (torch.device): The device on which the policy network runs.
        epsilon (float): Exploration rate for action selection.
        random_threshold (numpy.array): Array of random thresholds for epsilon-greedy action selection.

    Returns:
        torch.tensor: The probabilities for all actions.
    """

    if random_threshold[episode_index, agent_index] < epsilon:
        with torch.no_grad():
            output_size = policy_networks[0](state).size(0)
        probs = torch.tensor(np.random.rand(output_size), device=device)
    else:
        if nn_by_zone:
            probs = policy_networks[0](state)
        else:
            probs = policy_networks[agent_index](state)  # Greedy action
        
    return probs.detach()

def save_model(network, filename):
    """
    Saves the state dictionary of the given policy network to a file.

    Parameters:
        network (torch.nn.Module): The neural network to be saved.
        filename (str): The filename where the state dictionary will be saved.

    Returns:
        None
    """
    torch.save(network.state_dict(), filename)


