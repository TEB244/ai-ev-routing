import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

# Define the QNetwork architecture
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, layers):
        super(QNetwork, self).__init__()

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
        return self.output(x)

def initialize(state_dim, action_dim, layers, device_agents):

    """
    Initializes the Q-network and target Q-network for the DQN agent.

    Parameters:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        layers (list): List of integers defining the architecture of the neural networks.

    Returns:
        tuple: A tuple containing:
            - q_network (QNetwork): Initialized Q-network.
            - target_q_network (QNetwork): Initialized target Q-network with the same weights as the Q-network.
    """

    q_network = QNetwork(state_dim, action_dim, layers)  # Q-network
    target_q_network = QNetwork(state_dim, action_dim, layers)  # Target Q-network
    target_q_network.load_state_dict(q_network.state_dict())  # Initialize target Q-network with the same weights as Q-network
    return q_network.to(device_agents), target_q_network.to(device_agents)

def compute_loss(experiences, gamma, q_network, target_q_network):
    """
    Computes the loss for training the Q-network when optimizing all outputs in the Q-value distribution.

    Parameters:
        experiences (tuple): A tuple containing:
            - states (torch.tensor): Batch of states.
            - distributions (torch.tensor): Batch of action distributions.
            - rewards (torch.tensor): Batch of rewards.
            - next_states (torch.tensor): Batch of next states.
            - dones (torch.tensor): Batch of done flags indicating episode termination.
        gamma (float): Discount factor for future rewards.
        q_network (QNetwork): Q-network to be trained.
        target_q_network (QNetwork): Target Q-network used for computing target Q-values.

    Returns:
        torch.tensor: The computed loss value.
    """

    states, distributions, rewards, next_states, dones = experiences

    # Compute current Q-value predictions for all actions
    current_Q_values = q_network(states)

    # Compute the next Q-values for all actions from the target network
    with torch.no_grad():
        next_Q_values = target_q_network(next_states)
        # Get the maximum Q-value for each next state to use for each distribution element
        max_next_Q_values = next_Q_values.max(1, keepdim=True)[0]  # Max Q-value per next state, shaped as (batch_size, 1)

    # Compute target Q-values for each output in the distribution
    # We add (gamma * max_next_Q_values * (1 - dones)) to each element in the distribution to create per-action targets
    target_Q_values = rewards + (gamma * max_next_Q_values * (1 - dones))

    # Use a loss that can handle the entire Q-value distribution
    # Here we use MSE to compare each element in current_Q_values to target_Q_values
    loss = F.mse_loss(current_Q_values, target_Q_values.expand_as(current_Q_values))

    return loss

def agent_learn(experiences, gamma, q_network, target_q_network, optimizer, device):

    """
    Performs a learning step for the agent by computing the loss and updating the Q-network's weights.

    Parameters:
        experiences (tuple): A tuple containing:
            - states (numpy.array): Batch of states.
            - distributions (numpy.array): Batch of action distributions.
            - rewards (numpy.array): Batch of rewards.
            - next_states (numpy.array): Batch of next states.
            - dones (numpy.array): Batch of done flags indicating episode termination.
        gamma (float): Discount factor for future rewards.
        q_network (QNetwork): Q-network to be trained.
        target_q_network (QNetwork): Target Q-network used for computing target Q-values.
        optimizer (torch.optim.Optimizer): Optimizer for updating the Q-network's weights.

    Returns:
        None
    """

    # Convert NumPy arrays to PyTorch tensors
    states, distributions, rewards, next_states, dones = experiences
    states = torch.tensor(states, dtype=torch.float32, device=device)
    distributions = torch.tensor(distributions, dtype=torch.int64, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
    experiences = (states, distributions, rewards, next_states, dones)

    loss = compute_loss(experiences, gamma, q_network, target_q_network)  # Compute loss
    optimizer.zero_grad()  # Zero out gradients
    loss.backward()  # Backpropagate loss
    optimizer.step()  # Update weights

def get_actions(state, q_networks, random_threshold, epsilon, episode_index, agent_index, device, nn_by_zone):

    """
    Selects actions for an agent using an epsilon-greedy policy.

    Parameters:
        state (torch.tensor): The current state of the agent.
        q_networks (list): List of Q-networks for each agent.
        random_threshold (numpy.array): Array of random thresholds for epsilon-greedy action selection.
        epsilon (float): Exploration rate for epsilon-greedy policy.
        episode_index (int): Index of the current episode.
        agent_index (int): Index of the current agent.
        nn_by_zone (bool): True if using one neural network for each zone, and false if using a neural network for each car

    Returns:
        torch.tensor: The action values for the given state.
    """

    if random_threshold[episode_index, agent_index] < epsilon:  # Epsilon-greedy action selection
        if nn_by_zone:
            action_values = q_networks[0](state)
        else:
            action_values = q_networks[agent_index](state)
        noise = torch.randn(action_values.size()) * epsilon  # Match the size of the action_values tensor
        action_values += noise.to(device)  # Add noise for exploration
    else:
        if nn_by_zone:
            action_values = q_networks[0](state)  # Greedy action
        else:
            action_values = q_networks[agent_index](state)  # Greedy action

    return action_values.detach()

def soft_update(target_network, source_network, tau=0.001):

    """
    Performs a soft update of the target network's parameters using the source network's parameters.

    Parameters:
        target_network (torch.nn.Module): The target Q-network to be updated.
        source_network (torch.nn.Module): The source Q-network providing the parameters.
        tau (float, optional): The interpolation parameter (default is 0.001). Controls the update rate.

    Returns:
        None
    """

    for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

def save_model(network, filename):
    """
    Saves the state dictionary of the given neural network to a file.

    Parameters:
        network (torch.nn.Module): The neural network to be saved.
        filename (str): The filename where the state dictionary will be saved.

    Returns:
        None
    """

    torch.save(network.state_dict(), filename)
