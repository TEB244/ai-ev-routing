"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.

Part of the code was adapted from the following: 
* https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py
* https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py

Both are licensed under the MIT License.
"""

import torch
import torch.nn as nn
import random

import transformers
from transformers import GPT2Config

from ._transformer_backbone import GPT2Model
import math
import numpy as np
import torch.nn.functional as F
from training_processes.odt.odt_helpers.lamb import Lamb
from pathlib import Path
from torch import distributions as pyd


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    Squashed Normal Distribution(s)

    If loc/std is of size (batch_size, sequence length, d),
    this returns batch_size * sequence length * d
    independent squashed univariate normal distributions.
    """

    def __init__(self, loc, std):
        self.loc = loc
        self.std = std
        self.base_dist = pyd.Normal(loc, std)

        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self, N=1):
        # sample from the distribution and then compute
        # the empirical entropy:
        x = self.rsample((N,))
        log_p = self.log_prob(x)

        # log_p: (batch_size, context_len, action_dim),
        return -log_p.mean(axis=0).sum(axis=2)

    def log_likelihood(self, x):
        # log_prob(x): (batch_size, context_len, action_dim)
        # sum up along the action dimensions
        # Return tensor shape: (batch_size, context_len)
        return self.log_prob(x).sum(axis=2)


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, hidden_dim, act_dim, log_std_bounds=[-5.0, 2.0]):
        super().__init__()

        self.mu = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std_bounds = log_std_bounds

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.mu(obs), self.log_std(obs)
        log_std = torch.tanh(log_std)
        # log_std is the output of tanh so it will be between [-1, 1]
        # map it to be between [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
        std = log_std.exp()
        return SquashedNormal(mu, std)

class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_predictions(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(states[-1]), torch.zeros_like(actions[-1]), torch.zeros_like(rewards[-1])
        
class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        action_range,
        max_ep_len,
        n_positions,
        stochastic_policy,
        target_entropy,
        odt_config,
        action_tanh=True,
        **kwargs
    ):
        hidden_size=odt_config['embed_dim']
        ordering=odt_config['ordering']
        max_length=odt_config['K']
        eval_context_length=odt_config['eval_context_length']
        init_temperature=odt_config['init_temperature']
        resid_pdrop=odt_config['dropout']
        attn_pdrop=odt_config['dropout']
        n_layer=odt_config['n_layer']
        n_head=odt_config['n_head']
        n_inner=4 * odt_config['embed_dim']
        activation_function=odt_config['activation_function']
        
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = GPT2Config(
            vocab_size              = 1,                       
            n_positions             = max_ep_len,              
            n_embd                  = hidden_size,             
            n_layer                 = n_layer,                 
            n_head                  = n_head,                  
            resid_pdrop             = resid_pdrop,             
            attn_pdrop              = attn_pdrop,              
            activation_function     = activation_function,     
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        if ordering:
            self.embed_ordering = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        if stochastic_policy:
            self.predict_action = DiagGaussianActor(hidden_size, self.act_dim)
        else:
            self.predict_action = nn.Sequential(
                *(
                    [nn.Linear(hidden_size, self.act_dim)]
                    + ([nn.Tanh()] if action_tanh else [])
                )
            )
        self.stochastic_policy = stochastic_policy
        self.eval_context_length = eval_context_length
        self.ordering = ordering
        self.action_range = action_range

        if stochastic_policy:
            self.log_temperature = torch.tensor(np.log(init_temperature))
            self.log_temperature.requires_grad = True
            self.target_entropy = target_entropy

    def temperature(self):
        if self.stochastic_policy:
            return self.log_temperature.exp()
        else:
            return None

    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        ordering,
        padding_mask=None,
    ):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if padding_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            padding_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)

        if self.ordering:
            order_embeddings = self.embed_ordering(timesteps)
        else:
            order_embeddings = 0.0

        state_embeddings = state_embeddings + order_embeddings
        action_embeddings = action_embeddings + order_embeddings
        returns_embeddings = returns_embeddings + order_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_padding_mask = (
            torch.stack((padding_mask, padding_mask, padding_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_padding_mask,
        )
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # predict next return given state and action
        return_preds = self.predict_return(x[:, 2])
        # predict next state given state and action
        state_preds = self.predict_state(x[:, 2])
        # predict next action given state
        action_preds = self.predict_action(x[:, 1])

        return state_preds, action_preds, return_preds

    def get_predictions(
        self, states, actions, rewards, returns_to_go, timesteps, num_envs=1, **kwargs
    ):
        # we don't care about the past rewards in this model
        # tensor shape: batch_size, seq_length, variable_dim
        states = states.reshape(num_envs, -1, self.state_dim)
        actions = actions.reshape(num_envs, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(num_envs, -1, 1)

        # tensor shape: batch_size, seq_length
        timesteps = timesteps.reshape(num_envs, -1)

        # max_length is the DT context length (should be input length of the subsequence)
        # eval_context_length is the how long you want to use the history for your prediction
        if self.max_length is not None:
            states = states[:, -self.eval_context_length :]
            actions = actions[:, -self.eval_context_length :]
            returns_to_go = returns_to_go[:, -self.eval_context_length :]
            timesteps = timesteps[:, -self.eval_context_length :]

            ordering = torch.tile(
                torch.arange(timesteps.shape[1], device=states.device),
                (num_envs, 1),
            )
            # pad all tokens to sequence length
            padding_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            padding_mask = padding_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            padding_mask = padding_mask.repeat((num_envs, 1))

            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (
                            returns_to_go.shape[0],
                            self.max_length - returns_to_go.shape[1],
                            1,
                        ),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)

            ordering = torch.cat(
                [
                    torch.zeros(
                        (ordering.shape[0], self.max_length - ordering.shape[1]),
                        device=ordering.device,
                    ),
                    ordering,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            padding_mask = None
        state_preds, action_preds, return_preds = self.forward(
            states,
            actions,
            None,
            returns_to_go,
            timesteps,
            ordering,
            padding_mask=padding_mask,
            **kwargs
        )
        if self.stochastic_policy:
            return state_preds[:, -1], action_preds, return_preds[:, -1]
        else:
            return (
                state_preds[:, -1],
                self.clamp_action(action_preds[:, -1]),
                return_preds[:, -1],
            )

    def clamp_action(self, action):
        return action.clamp(*self.action_range)

    def set_optimizer_scheduler(self, config):
        self.optimizer = Lamb(self.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'], eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda steps: min((steps + 1) / config['warmup_steps'], 1))
        self.log_temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=1e-4, betas=[0.9, 0.999])

    def _save_weights(self, path_prefix, is_offline_model=False):
        to_save = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
        }
        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\n Model saved at {path_prefix}/model.pt")
        if is_offline_model:
            with open(f"{path_prefix}/offline_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/offline_model.pt")

    def _load_weights(self, path_prefix, load_optimizer=True):
        model_file = Path(f"{path_prefix}/model.pt")
        if not model_file.exists():
            return
        with open(model_file, "rb") as f:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                map_location = lambda storage, loc: storage.cuda(0)
            else:
                map_location = torch.device("cpu")
            checkpoint = torch.load(f, map_location=map_location, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
    
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(
                checkpoint["log_temperature_optimizer_state_dict"]
            )
        np.random.set_state(checkpoint["np"])
        random.setstate(checkpoint["python"])
    
        pytorch_rng_state = checkpoint.get("pytorch")
        if pytorch_rng_state is not None:
            if not isinstance(pytorch_rng_state, torch.ByteTensor):
                pytorch_rng_state = torch.tensor(
                    pytorch_rng_state, dtype=torch.uint8, device="cpu"
                )
            torch.set_rng_state(pytorch_rng_state)
        print(f"Model loaded at {model_file}")
        
    def get_attn_layers(self, device):
        attn_number = len(self.transformer.h)
        attn_layer_size = self.transformer.h[0].attn.c_attn.weight.shape
        attn_layers = torch.empty((attn_number, attn_layer_size[0], attn_layer_size[1]), dtype=torch.float32, device=device)
        for i in range(attn_number):
            attn_layers[i, :, :] = self.transformer.h[i].attn.c_attn.weight
        return attn_layers
        
    def load_attn_layers(self, save_global_path):
        weights_path = f'{save_global_path}/global_weights.pth'
        if Path(weights_path).exists():
            global_weights = torch.load(weights_path)
            if isinstance(global_weights, list):
                global_weights_dict = {}
                for i, weights_dict in enumerate(global_weights):
                    if isinstance(weights_dict, dict):
                        for key, value in weights_dict.items():
                            global_weights_dict[f"zone_{i}_{key}"] = value
                    else:
                        raise ValueError("Expected a list of dictionaries, but found a different structure.")
                global_weights = global_weights_dict
    
            return global_weights
        else:
            raise FileNotFoundError(f"No global weights found at {weights_path}")

    def load_attn_layers(self, save_global_path):
        weights_path = Path(save_global_path) / "global_weights.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"No global weights found at {weights_path}")

        raw = torch.load(weights_path, map_location="cpu", weights_only=False)
        if not isinstance(raw, torch.Tensor):
            raise ValueError(f"Expected a Tensor but found {type(raw)} in {weights_path}")

        num_layers = len(self.transformer.h)
        if raw.ndim != 3 or raw.shape[0] != num_layers:
            raise ValueError(
                f"Tensor must be [num_layers={num_layers}, D1, D2], "
                f"but got shape {tuple(raw.shape)}"
            )

        for i in range(num_layers):
            w = raw[i].to(self.transformer.h[i].attn.c_attn.weight.device)
            self.transformer.h[i].attn.c_attn.weight = torch.nn.Parameter(w)

        # Reset scheduler epoch 
        if hasattr(self, "scheduler") and self.scheduler is not None:
            self.scheduler.last_epoch = -1