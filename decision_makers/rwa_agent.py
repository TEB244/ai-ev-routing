
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with multi-head self-attention and feed-forward network."""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x):
        # Pre-norm self-attention with residual
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # Pre-norm FFN with residual
        normed = self.ln2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x


class RWANetwork(nn.Module):
    """
    Hybrid MLP + Per-Charger Cross-Attention policy network for the RWA algorithm.

    Architecture (v4 - per-charger queries):
        1. MLP backbone processes the FULL state vector (same as REINFORCE) to guarantee
           a learning floor — the network can learn at least as well as a plain MLP.
        2. Per-charger learned queries cross-attend to charger-leg tokens. Each physical
           charger gets its own query conditioned on the backbone output, producing a
           charger-specific attended representation. This allows the attention to learn
           different patterns for different chargers (e.g., "charger A should attend to
           nearby alternatives" vs "charger B should attend to low-traffic options").
        3. Shared per-charger head maps each charger's attended output to 3 action dims
           (the routing weights for that charger's 3 legs).
        4. Backbone produces a global action prediction (learning floor). Attention output
           is added as a gated residual — preserving the MLP learning floor while allowing
           attention to provide structured per-charger adjustments.

    The action space is continuous [0,1] per charger-leg dimension. Sigmoid is applied
    externally (in the training loop) to map logits to [0,1] for the environment.
    A learnable log-standard-deviation parameter controls exploration.

    This design ensures:
        - Guaranteed learning (backbone_head works even if attention contributes nothing)
        - Per-charger attention produces action-specific representations (not one global summary)
        - Attention benefit grows with more chargers (more tokens + more queries)
    """
    def __init__(self, state_dim, action_dim, layers, embed_dim=64, num_heads=4,
                 attention_dropout=0.0, num_transformer_layers=1):
        super(RWANetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_charger_tokens = (state_dim - 6) // 2  # Number of charger-leg tokens
        self.num_physical_chargers = action_dim // 3     # Number of physical chargers

        # --- MLP backbone (processes full state, same structure as REINFORCE) ---
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
        )
        backbone_dim = layers[1]
        self.backbone_dim = backbone_dim

        # --- Charger token embedding ---
        self.charger_embed = nn.Linear(2, backbone_dim)

        # --- Per-charger learned queries ---
        self.query_tokens = nn.Parameter(
            torch.randn(self.num_physical_chargers, backbone_dim) * 0.02
        )
        self.query_context = nn.Linear(backbone_dim, backbone_dim)

        # --- Cross-attention ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=backbone_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attn_ln = nn.LayerNorm(backbone_dim)

        # --- Per-charger action head (shared weights across chargers) ---
        self.per_charger_head = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, 3),
        )

        # --- Backbone global action prediction (learning floor, like REINFORCE) ---
        self.backbone_head = nn.Sequential(
            nn.Linear(backbone_dim, layers[-1]),
            nn.ReLU(),
            nn.Linear(layers[-1], action_dim),
        )

        # --- Learnable gate for attention contribution ---
        self.gate = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5 initially

        # Learnable log-standard-deviation for the Gaussian policy (one per action dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, state):
        """
        Forward pass through the hybrid MLP + per-charger cross-attention network.

        Parameters:
            state (torch.Tensor): Input state tensor of shape (state_dim,) or (batch, state_dim).

        Returns:
            torch.Tensor: Action means (raw logits) of shape (action_dim,) or (batch, action_dim).
        """
        squeezed = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeezed = True

        batch_size = state.shape[0]

        # --- MLP backbone on full state ---
        backbone_out = self.backbone(state)  # (B, backbone_dim)

        # --- Parse and embed charger tokens ---
        # State layout is [t0, t1, ..., tN, d0, d1, ..., dN, global_context]
        # (all traffic values first, then all distances — NOT interleaved)
        num_t = self.num_charger_tokens
        traffic = state[:, :num_t]                    # (B, num_legs)
        distances = state[:, num_t:num_t * 2]         # (B, num_legs)
        charger_features = torch.stack([traffic, distances], dim=-1)  # (B, num_legs, 2)
        charger_tokens = self.charger_embed(charger_features)  # (B, N_legs, backbone_dim)

        # --- Per-charger queries conditioned on backbone ---
        context = self.query_context(backbone_out).unsqueeze(1)  # (B, 1, backbone_dim)
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1) + context
        # queries: (B, num_physical_chargers, backbone_dim)

        # --- Cross-attention: per-charger queries attend to all charger-leg tokens ---
        attn_out, _ = self.cross_attn(queries, charger_tokens, charger_tokens)
        # attn_out: (B, num_physical_chargers, backbone_dim)
        attn_out = self.attn_ln(attn_out)

        # --- Per-charger actions from attention ---
        per_charger_action = self.per_charger_head(attn_out)  # (B, num_physical_chargers, 3)
        attn_action = per_charger_action.reshape(batch_size, -1)  # (B, action_dim)

        # --- Backbone global action prediction ---
        backbone_action = self.backbone_head(backbone_out)  # (B, action_dim)

        # --- Gated residual fusion ---
        gate = torch.sigmoid(self.gate)
        mean = backbone_action + gate * attn_action

        if squeezed:
            mean = mean.squeeze(0)

        return mean

    def get_distribution(self, state):
        """
        Returns a Gaussian distribution over actions for the given state.

        Parameters:
            state (torch.Tensor): Input state.

        Returns:
            torch.distributions.Normal: Gaussian distribution with learned mean and std.
        """
        mean = self.forward(state)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)


def initialize(state_dim, action_dim, layers, device_agents, embed_dim=64, num_heads=4, attention_dropout=0.0):
    """
    Initializes the RWANetwork for the RWA agent.

    Parameters:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        layers (list): List of integers defining the architecture of the neural networks.
            layers[0] is the backbone hidden dim (e.g., 128), layers[1] is used for
            the backbone output dim and policy head hidden dim (e.g., 64).
        device_agents (torch.device): The device to which the network will be moved.
        embed_dim (int): Embedding dimension for cross-attention (defaults to layers[1]).
        num_heads (int): Number of attention heads.
        attention_dropout (float): Dropout rate for attention layers.

    Returns:
        rwa_network (RWANetwork): Initialized RWANetwork.
    """
    rwa_network = RWANetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        layers=layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        attention_dropout=attention_dropout,
        num_transformer_layers=1,
    )
    return rwa_network.to(device_agents)


def compute_loss(experiences, gamma, rwa_network):
    """
    Computes the continuous policy gradient loss for the RWA agent.

    Uses a Gaussian policy: the network outputs action means, and log-probabilities
    are computed under Normal(mean, std) for the actual continuous actions taken.
    The architectural advantage comes from the attention network itself, not from
    loss function modifications. Includes return normalization and entropy bonus.

    Parameters:
        experiences (tuple): A tuple containing:
            - states (torch.tensor): Batch of states.
            - actions (torch.tensor): Batch of continuous actions taken (logit-space values).
            - rewards (torch.tensor): Batch of rewards.
            - dones (torch.tensor): Batch of done flags indicating episode termination.
        gamma (float): Discount factor for future rewards.
        rwa_network (RWANetwork): The RWA network to be trained.

    Returns:
        torch.tensor: The computed loss value.
    """
    states, actions, rewards, dones = experiences

    # Compute discounted returns for each timestep
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
    dist = rwa_network.get_distribution(states)
    log_probs = dist.log_prob(actions).sum(dim=-1)  # Sum log-probs across action dimensions

    # Entropy bonus to encourage exploration (prevents premature convergence)
    entropy = dist.entropy().sum(dim=-1)

    # REINFORCE loss: -(log_prob * return) with entropy bonus
    loss = -(log_probs * returns).mean() - 0.01 * entropy.mean()

    return loss


def agent_learn(experiences, gamma, rwa_network, optimizer, device):
    """
    Performs a learning step for the RWA agent by computing the loss and updating the network's weights.

    Parameters:
        experiences (tuple): A tuple containing:
            - states (torch.tensor): Batch of states.
            - actions (torch.tensor): Batch of continuous actions taken.
            - rewards (torch.tensor): Batch of rewards.
            - dones (torch.tensor): Batch of done flags indicating episode termination.
        gamma (float): Discount factor for future rewards.
        rwa_network (RWANetwork): The RWA network to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating the network's weights.
        device (torch.device): The device to which the tensors will be moved.

    Returns:
        None
    """
    loss = compute_loss(experiences, gamma, rwa_network)

    optimizer.zero_grad()
    loss.backward()
    # Apply gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(rwa_network.parameters(), max_norm=1.0)
    optimizer.step()


def get_actions(state, rwa_networks, episode_index, agent_index, device, epsilon, random_threshold, nn_by_zone):
    """
    Selects continuous actions by sampling from the Gaussian policy.

    During exploration (epsilon-greedy), samples with increased noise.
    During exploitation, samples from the learned Gaussian distribution.
    Actions are returned in logit space; sigmoid is applied externally in the training loop
    before passing to the environment.

    Parameters:
        state (torch.tensor): The current state of the agent.
        rwa_networks (list): List of RWA networks for each agent.
        episode_index (int): Index of the current episode.
        agent_index (int): Index of the current agent.
        device (torch.device): The device on which the network runs.
        epsilon (float): Exploration rate for action selection.
        random_threshold (numpy.array): Array of random thresholds for epsilon-greedy action selection.
        nn_by_zone (bool): Whether to use a single network per zone.

    Returns:
        torch.tensor: Sampled continuous actions in logit space.
    """
    if nn_by_zone:
        net = rwa_networks[0]
    else:
        net = rwa_networks[agent_index]

    with torch.no_grad():
        if random_threshold[episode_index, agent_index] < epsilon:
            # Exploration: sample from policy with extra noise for broader search
            dist = net.get_distribution(state)
            explore_std = torch.exp(net.log_std) + 0.5  # Augmented standard deviation
            explore_dist = torch.distributions.Normal(dist.loc, explore_std)
            action = explore_dist.sample()
        else:
            # Exploitation: sample from learned Gaussian policy
            dist = net.get_distribution(state)
            action = dist.sample()

    return action.detach()


def save_model(network, filename):
    """
    Saves the state dictionary of the given RWA network to a file.

    Parameters:
        network (torch.nn.Module): The neural network to be saved.
        filename (str): The filename where the state dictionary will be saved.

    Returns:
        None
    """
    torch.save(network.state_dict(), filename)
