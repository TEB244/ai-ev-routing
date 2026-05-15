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
    Standard DQN loss: regress Q(s, a) for the action actually taken
    toward r + gamma * max_a' Q_target(s', a') * (1 - done).

    The stored "actions" are the raw Q-network outputs at action-collection
    time (the env consumes the sigmoid'd full 3-vector as continuous
    edge-weights). We take argmax over those stored output vectors as a
    discrete proxy for "which action was emphasized," and update only the
    Q-value of that dimension - matching the conventional DQN update where
    only Q(s, a_taken) is shifted toward the bootstrap target.

    Prior implementation broadcast the target to all action dimensions,
    which discarded the action information entirely and trained every
    output to predict the same value - preventing any preference between
    actions from being learned.

    Parameters:
        experiences (tuple):
            - states         (torch.tensor): batch of states, shape (B, state_dim)
            - action_outputs (torch.tensor): batch of stored Q-net outputs at
                                             action-collection time, shape (B, action_dim)
            - rewards        (torch.tensor): batch of rewards, shape (B, 1)
            - next_states    (torch.tensor): batch of next states, shape (B, state_dim)
            - dones          (torch.tensor): batch of done flags, shape (B, 1)
        gamma (float): discount factor for future rewards.
        q_network (QNetwork): Q-network being trained.
        target_q_network (QNetwork): Target Q-network for the bootstrap.

    Returns:
        torch.tensor: scalar MSE loss.
    """

    states, action_outputs, rewards, next_states, dones = experiences

    # Q-values for every action in the current state (B, action_dim).
    current_Q_values = q_network(states)

    # Bootstrap target: r + gamma * max_a' Q_target(s', a') * (1 - done).
    # Shape (B, 1) -> (B,).
    with torch.no_grad():
        next_Q_values = target_q_network(next_states)
        max_next_Q_values = next_Q_values.max(1, keepdim=True)[0]  # (B, 1)
    target_Q = (rewards + gamma * max_next_Q_values * (1 - dones)).squeeze(1)  # (B,)

    # Action selected at collection time: argmax over the stored Q-net output
    # vector. Sigmoid is monotonic, so argmax(action_outputs) == argmax(sigmoid(action_outputs)).
    chosen_action_idx = action_outputs.argmax(dim=-1, keepdim=True)        # (B, 1)
    chosen_Q = current_Q_values.gather(1, chosen_action_idx).squeeze(1)    # (B,)

    # Standard DQN MSE on the chosen action's Q-value only.
    loss = F.mse_loss(chosen_Q, target_Q)

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

    # Convert NumPy arrays / tensors to PyTorch tensors on the right device.
    # CRITICAL: action_outputs must stay as float32 - the previous int64 cast
    # truncated the stored Q-value outputs to integers, destroying the action
    # information that compute_loss now relies on.
    states, action_outputs, rewards, next_states, dones = experiences
    states         = torch.as_tensor(states,         dtype=torch.float32, device=device)
    action_outputs = torch.as_tensor(action_outputs, dtype=torch.float32, device=device)
    rewards        = torch.as_tensor(rewards,        dtype=torch.float32, device=device).unsqueeze(1)
    next_states    = torch.as_tensor(next_states,    dtype=torch.float32, device=device)
    dones          = torch.as_tensor(dones,          dtype=torch.float32, device=device).unsqueeze(1)
    experiences    = (states, action_outputs, rewards, next_states, dones)

    loss = compute_loss(experiences, gamma, q_network, target_q_network)  # Compute loss
    optimizer.zero_grad()  # Zero out gradients
    loss.backward()  # Backpropagate loss
    # Gradient clipping prevents pathological updates when bootstrap targets
    # are large in magnitude (rewards are in [-700, -40] for this env).
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
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
