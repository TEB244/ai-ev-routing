
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    """
    A neural network for the REINFORCE algorithm that outputs continuous actions
    via a Gaussian policy. The network outputs action means (logits), and a learnable
    log-standard-deviation parameter controls exploration.

    The action space is continuous [0,1] per charger-leg dimension, where each dimension
    independently controls the distance vs. traffic tradeoff weight for graph reweighting.
    Sigmoid is applied externally (in the training loop) to map logits to [0,1] for the environment.
    """
    LOG_STD_MIN = -2.0  # std floor ≈ 0.135 — prevents deterministic collapse
    LOG_STD_MAX = 0.5   # std ceiling ≈ 1.65 — prevents excessive noise

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

        # Learnable log-standard-deviation for the Gaussian policy (one per action dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))  # std≈0.6 instead of 1.0 for less initial noise

    def forward(self, state):
        x = state
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = torch.relu(x)  # Apply ReLU activation
        mean = self.output(x)
        return mean  # Raw logits (action means), no softmax

    def get_distribution(self, state):
        """
        Returns a Gaussian distribution over actions for the given state.

        Parameters:
            state (torch.Tensor): Input state.

        Returns:
            torch.distributions.Normal: Gaussian distribution with learned mean and std.
        """
        mean = self.forward(state)
        clamped_log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(clamped_log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

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
    Computes the continuous policy gradient loss for the REINFORCE agent.

    Uses a Gaussian policy: the network outputs action means, and log-probabilities
    are computed under Normal(mean, std) for the actual continuous actions taken.
    Includes return normalization for variance reduction and an entropy bonus
    to encourage exploration.

    Parameters:
        experiences (tuple): A tuple containing:
            - states (torch.tensor): Batch of states.
            - actions (torch.tensor): Batch of continuous actions taken (logit-space values).
            - rewards (torch.tensor): Batch of rewards.
            - dones (torch.tensor): Batch of done flags indicating episode termination.
        gamma (float): Discount factor for future rewards.
        policy_network (PolicyNetwork): The policy network to be trained.

    Returns:
        torch.tensor: The computed policy gradient loss value.
    """

    states, actions, rewards, dones = experiences

    # Compute discounted returns for each time step
    returns = []
    G = 0
    for r, done in zip(reversed(rewards), reversed(dones)):
        G = r + gamma * G * (1 - done)
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32, device=states.device)

    # Normalize returns for variance reduction (baseline-free variance reduction)
    if len(returns) > 2:  # Skip normalization for <=2 points (produces pure noise)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Get Gaussian distribution and compute log-probability of taken actions
    dist = policy_network.get_distribution(states)
    log_probs = dist.log_prob(actions).sum(dim=-1)  # Sum log-probs across action dimensions

    # Entropy bonus to encourage exploration (prevents premature convergence)
    entropy = dist.entropy().sum(dim=-1)

    # REINFORCE loss: -(log_prob * return) with entropy bonus
    loss = -(log_probs * returns).mean() - 0.02 * entropy.mean()
    return loss

def agent_learn(experiences, gamma, policy_network, optimizer, device):
    """
    Performs a learning step for the REINFORCE agent by computing the loss and updating the policy network's weights.

    Parameters:
        experiences (tuple): A tuple containing:
            - states (torch.tensor): Batch of states.
            - actions (torch.tensor): Batch of continuous actions taken.
            - rewards (torch.tensor): Batch of rewards.
            - dones (torch.tensor): Batch of done flags indicating episode termination.
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
    Selects continuous actions by sampling from the Gaussian policy.

    During exploration (epsilon-greedy), samples with increased noise.
    During exploitation, samples from the learned Gaussian distribution.
    Actions are returned in logit space; sigmoid is applied externally in the training loop
    before passing to the environment.

    Parameters:
        state (torch.tensor): The current state of the agent.
        policy_networks (list): List of policy networks for each agent.
        episode_index (int): Index of the current episode.
        agent_index (int): Index of the current agent.
        device (torch.device): The device on which the policy network runs.
        epsilon (float): Exploration rate for action selection.
        random_threshold (numpy.array): Array of random thresholds for epsilon-greedy action selection.
        nn_by_zone (bool): Whether to use a single network per zone.

    Returns:
        torch.tensor: Sampled continuous actions in logit space.
    """

    if nn_by_zone:
        net = policy_networks[0]
    else:
        net = policy_networks[agent_index]

    with torch.no_grad():
        if random_threshold[episode_index, agent_index] < epsilon:
            # Exploration: sample from policy with extra noise for broader search
            dist = net.get_distribution(state)
            clamped_log_std = torch.clamp(net.log_std, net.LOG_STD_MIN, net.LOG_STD_MAX)
            explore_std = torch.exp(clamped_log_std) + 0.5  # Augmented standard deviation
            explore_dist = torch.distributions.Normal(dist.loc, explore_std)
            action = explore_dist.sample()
        else:
            # Exploitation: sample from learned Gaussian policy
            dist = net.get_distribution(state)
            action = dist.sample()

    return action.detach()

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
