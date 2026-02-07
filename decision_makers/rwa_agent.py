
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with multi-head self-attention and feed-forward network."""
    def __init__(self, embed_dim, num_heads, attention_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(attention_dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(attention_dropout),
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
    Attention-based policy network for the RWA (RL with Attention) algorithm.

    Parses the flat state vector into per-charger tokens (traffic, distance) and
    global context features, then applies multi-head self-attention so the network
    can learn which chargers to focus on given the current traffic/distance landscape.

    Architecture:
        state → parse into charger tokens + context token
        → embed → positional encoding → N transformer blocks
        → extract context token → MLP policy head → softmax → action probs
    """
    def __init__(self, state_dim, action_dim, layers, embed_dim=128, num_heads=4,
                 attention_dropout=0.1, num_transformer_layers=2):
        super(RWANetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_tokens = (state_dim - 6) // 2  # Number of charger-leg tokens
        self.embed_dim = embed_dim

        # Embedding layers
        self.charger_embed = nn.Linear(2, embed_dim)          # Per-charger (traffic, distance) → embed_dim
        self.global_embed = nn.Linear(6, embed_dim)            # Global context features → embed_dim (CLS token)

        # Positional encoding for up to 31 tokens (1 context + 30 charger-legs = 10 chargers × 3 legs)
        max_positions = 31
        self.pos_embed = nn.Embedding(max_positions, embed_dim)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, attention_dropout)
            for _ in range(num_transformer_layers)
        ])

        # Final layer norm after transformer stack
        self.final_ln = nn.LayerNorm(embed_dim)

        # Policy head: context token → action probabilities
        head_hidden = layers[1] if len(layers) > 1 else 64
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, action_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, state):
        """
        Forward pass through the attention-based network.

        Parameters:
            state (torch.Tensor): Input state tensor of shape (state_dim,) or (batch, state_dim).

        Returns:
            torch.Tensor: Action probabilities of shape (action_dim,) or (batch, action_dim).
        """
        squeezed = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeezed = True

        batch_size = state.shape[0]
        num_tokens = self.num_tokens

        # Parse state into charger features and global features
        charger_features = state[:, :num_tokens * 2].reshape(batch_size, num_tokens, 2)
        global_features = state[:, num_tokens * 2:]

        # Embed charger tokens and global context
        charger_embeds = self.charger_embed(charger_features)       # (B, N, embed_dim)
        context_embed = self.global_embed(global_features)          # (B, embed_dim)
        context_embed = context_embed.unsqueeze(1)                  # (B, 1, embed_dim)

        # Concatenate: [context_token, charger_token_0, charger_token_1, ...]
        tokens = torch.cat([context_embed, charger_embeds], dim=1)  # (B, N+1, embed_dim)

        # Add positional encoding
        seq_len = tokens.size(1)
        positions = torch.arange(seq_len, device=state.device)
        tokens = tokens + self.pos_embed(positions)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens)

        # Final layer norm
        tokens = self.final_ln(tokens)

        # Pool: extract context token (position 0) — CLS-style
        pooled = tokens[:, 0, :]  # (B, embed_dim)

        # Policy head → softmax
        logits = self.policy_head(pooled)
        probs = F.softmax(logits, dim=-1)

        if squeezed:
            probs = probs.squeeze(0)

        return probs


def initialize(state_dim, action_dim, layers, device_agents, embed_dim=128, num_heads=4, attention_dropout=0.1):
    """
    Initializes the RWANetwork for the RWA agent.

    Parameters:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        layers (list): List of integers defining the architecture of the neural networks.
        device_agents (torch.device): The device to which the network will be moved.
        embed_dim (int): Embedding dimension for attention layers.
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
        num_transformer_layers=2,
    )
    return rwa_network.to(device_agents)


def compute_loss(experiences, gamma, rwa_network):
    """
    Computes the loss for the RWA agent using policy gradients with baseline subtraction
    and an entropy bonus for exploration.

    Improvements over vanilla REINFORCE:
        1. Return normalization (baseline subtraction) to reduce gradient variance
        2. Entropy bonus to prevent premature policy collapse

    Parameters:
        experiences (tuple): A tuple containing:
            - states (torch.tensor): Batch of states.
            - actions (torch.tensor): Batch of actions (or action distributions).
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

    # Baseline subtraction: normalize returns to reduce variance
    if returns.numel() > 1 and returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Get action probabilities from attention-based network
    probs = rwa_network(states)

    # Convert action distributions to discrete action indices
    action_indices = torch.argmax(actions, dim=-1)

    # Compute log probabilities for chosen actions
    log_probs = torch.log(torch.gather(probs, 1, action_indices.unsqueeze(1)).squeeze(1) + 1e-8)

    # Policy gradient loss with baseline
    policy_loss = -(log_probs * returns).mean()

    # Entropy bonus: encourages exploration by penalizing low-entropy (deterministic) policies
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
    entropy_coeff = 0.01

    loss = policy_loss - entropy_coeff * entropy

    return loss


def agent_learn(experiences, gamma, rwa_network, optimizer, device):
    """
    Performs a learning step for the RWA agent by computing the loss and updating the network's weights.

    Parameters:
        experiences (tuple): A tuple containing:
            - states (numpy.array): Batch of states.
            - actions (numpy.array): Batch of actions taken.
            - rewards (numpy.array): Batch of rewards.
            - dones (numpy.array): Batch of done flags indicating episode termination.
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
    torch.nn.utils.clip_grad_norm_(rwa_network.parameters(), max_norm=1.0)
    optimizer.step()


def get_actions(state, rwa_networks, episode_index, agent_index, device, epsilon, random_threshold, nn_by_zone):
    """
    Selects actions for an agent using the attention-based policy network with epsilon-greedy exploration.

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
        torch.tensor: The probabilities for all actions.
    """
    if random_threshold[episode_index, agent_index] < epsilon:
        with torch.no_grad():
            output_size = rwa_networks[0](state).size(0)
        probs = torch.tensor(np.random.rand(output_size), device=device)
    else:
        if nn_by_zone:
            probs = rwa_networks[0](state)
        else:
            probs = rwa_networks[agent_index](state)

    return probs.detach()


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
