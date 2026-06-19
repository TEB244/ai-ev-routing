"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""
import sys

from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle
import random
import time
import torch
import numpy as np
import re
import os
import json
import glob
import pandas as pd
import h5py

from collections import defaultdict
from environment.evaluation import evaluate
from misc import utils
from misc.replay_buffer import ReplayBuffer
from misc.lamb import Lamb
from pathlib import Path
from misc.data import create_dataloader, TransformSamplingSubTraj
from decision_transformer.models.decision_transformer import DecisionTransformer
from misc.evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from misc.trainer import SequenceTrainer
from misc.logger import Logger
from misc.online_data import PersistentOnlineDataset, create_online_dataloader



MAX_EPISODE_LEN = 1000

class Experiment:
    def __init__(
        self,
        variant,
        environment,
        chargers,
        routes,
        state_dim,
        action_dim,
        device,
        seed,
        experiment_number,
        agg_num,
        zone_index,
        prev_trajs,
        metrics_path,
        ev_info,
        date,
        verbose,
        num_episodes,
        arwt
    ):
        # Basic attributes
        self.action_range = self._get_env_spec(variant)
        self.variant = variant
        self.environment = environment
        self.chargers = chargers
        self.routes = routes
        self.state_dim = state_dim
        self.act_dim = action_dim
        self.device = device
        self.seed = seed
        self.experiment_number = experiment_number
        self.agg_num = agg_num
        self.zone_index = zone_index
        self.ev_info = ev_info
        self.date = date
        self.verbose = verbose
        self.arwt = arwt
        # Store metrics path
        self.metrics_base_path = metrics_path


        # Setup save path and logger
        base_dir = variant["save_dir"]
        if variant.get("evaluation", False):
            base_dir = os.path.join(base_dir, "eval")
        self.save_dir = os.path.join(base_dir, str(variant["exp_name"]))
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = Logger(variant, agg_num, zone_index)

        # Load or persist normalization stats and initial trajectories
        stats_path = os.path.join(self.save_dir, 'state_stats.pkl')
        if agg_num < 1:
            # First aggregation: load offline trajectories and compute stats
            trajs, self.state_mean, self.state_std = self._load_dataset('merl')
            # Save stats for later
            with open(stats_path, 'wb') as f:
                pickle.dump({'state_mean': self.state_mean, 'state_std': self.state_std}, f)
        else:
            # Subsequent aggregation: reuse previous trajectories
            trajs = prev_trajs
            # Load saved stats
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
            self.state_mean, self.state_std = stats['state_mean'], stats['state_std']

        self.offline_trajs = trajs
        # Initialize replay buffer with chosen trajectories
        self.replay_buffer = ReplayBuffer(variant['replay_size'], trajs)

        # Build online dataset from replay buffer
        transform = TransformSamplingSubTraj(
            max_len=variant['K'],
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            state_mean=self.state_mean,
            state_std=self.state_std,
            reward_scale=1,
            action_range=self._get_env_spec(variant)
        )
        self.online_dataset = PersistentOnlineDataset(
            initial_trajectories=self.replay_buffer.trajectories,
            sample_size=variant['batch_size'] * variant['num_updates_per_online_iter'],
            transform=transform
        )
        self.online_dataloader = create_online_dataloader(
            self.online_dataset,
            batch_size=variant['batch_size']
        )

        # Initialize transformer model
        self.model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self._get_env_spec(variant),
            max_length=variant['K'],
            eval_context_length=variant['eval_context_length'],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4 * variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            stochastic_policy=True,
            ordering=variant['ordering'],
            init_temperature=variant['init_temperature'],
            target_entropy=-self.act_dim,
        ).to(device=self.device)

        # Evaluation-specific: Loads model from a different zone
        if variant.get('evaluation', False) and agg_num == 0:
            num_zones = variant.get('num_zones', 4)
            next_zone = (self.zone_index + 1) % num_zones
            model_path = (
                f"../exp/Exp_{self.experiment_number}"
                f"/Agg:0-Zone:{next_zone+1}/model.pt"
            )
            print(f"[eval] first-agg load from zone {next_zone}: {model_path}")
            self._load_model(model_path)
            self.save_attn_layers(self.model, self.device)

        # Setup optimizers and scheduler
        self.optimizer = Lamb(self.model.parameters(), lr=variant['learning_rate'], weight_decay=variant['weight_decay'], eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps + 1) / variant['warmup_steps'], 1)
        )
        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999]
        )

        # Load previous aggregation’s model if needed
        if agg_num > 0:
            prev_log = re.sub(r"Agg:\\d+", f"Agg:{agg_num-1}", self.logger.log_path)
            print(f"[federated] loading local checkpoint from {prev_log}")
            # load model + counters/RNG, skip optimizer
            self._load_model(prev_log, load_optimizer=False)
        
            global_path = f"saved_networks/Exp_{self.experiment_number}"
            print(f"[federated] applying aggregated weights from {global_path}")
            attn_layers = load_global_weights(global_path)
            self.set_attn_layers(self.model, attn_layers.to(self.device))
            #Reset LR scheduler epoch counter:
            self.scheduler.last_epoch = -1
        
            #Zero out any momentum buffers in the optimizer:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    state = self.optimizer.state[p]
                    if 'momentum_buffer' in state:
                        state['momentum_buffer'].zero_()

        # Tracking variables
        self.aug_trajs = []
        self.metrics = []
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        self.reward_scale = 1

    def _get_env_spec(self, variant):
        action_range = [
            1e-6,
            1e-6,
        ]
        return action_range

    def _save_model(self, path_prefix, is_pretrain_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
        }

        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\n Model saved at {path_prefix}/model.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")
        self.model_path = f"{path_prefix}/model.pt"

    # in train_odt.py, update this definition:
    def _load_model(self, path_prefix, load_optimizer=True):
        model_file = Path(f"{path_prefix}/model.pt")
        if not model_file.exists():
            return
    
        # 1) load checkpoint onto cuda:0 or CPU
        with open(model_file, "rb") as f:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                map_location = lambda storage, loc: storage.cuda(0)
            else:
                map_location = torch.device("cpu")
            checkpoint = torch.load(f, map_location=map_location)
    
        # 2) always restore model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
    
        # 3) restore optimizer & scheduler only if requested
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(
                checkpoint["log_temperature_optimizer_state_dict"]
            )
    
        # 4) always restore counters
        self.pretrain_iter = checkpoint["pretrain_iter"]
        self.online_iter = checkpoint["online_iter"]
        self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
    
        # 5) always restore numpy & python RNG
        np.random.set_state(checkpoint["np"])
        random.setstate(checkpoint["python"])
    
        # 6) fix & restore PyTorch RNG
        pytorch_rng_state = checkpoint.get("pytorch")
        if pytorch_rng_state is not None:
            if not isinstance(pytorch_rng_state, torch.ByteTensor):
                pytorch_rng_state = torch.tensor(
                    pytorch_rng_state, dtype=torch.uint8, device="cpu"
                )
            torch.set_rng_state(pytorch_rng_state)
    
        print(f"Model loaded at {model_file}")

    def _load_dataset(self, env_name):
        """
        Loads dataset from HDF5 (.h5) files instead of .pkl.

        Parameters:
        - env_name (str): Environment name (used for dataset location path).

        Returns:
        - trajectories (list): List of trajectory dictionaries.
        - state_mean (np.array): Mean values for state normalization.
        - state_std (np.array): Standard deviation values for state normalization.
        """
        # determine which zone to actually load from (wrap around)
        num_zones = self.variant.get("num_zones", 4)
        load_zone = (self.zone_index + 1) % num_zones
        print(f'Loading Dataset for Zone {load_zone} (orig {self.zone_index})...')

        # Locate the dataset file
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        adjusted_experiment_number = str(int(self.experiment_number) - 108)
        #dataset_path = f"/storage_1/epigou_storage/Exp_3000/Exp_4000/data_zone_{self.zone_index + 1}.h5"


        dataset_path = os.path.join(base_dir, f'rl-for-vrp-csp/Exp_3000/data_zone_{load_zone}.h5')

        # dataset_path = os.path.join(
        #     base_dir,
        #     f'rl-for-vrp-csp/Exp_{adjusted_experiment_number}/data_zone_{load_zone}.h5'
        # )

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"No .h5 files found for zone {load_zone} in {data_dir}. Exiting...")
            exit(1)

        # Load trajectories from HDF5 file
        trajectories = []
        try:
            with h5py.File(dataset_path, 'r') as f:
                zone_key = f"zone_{load_zone}"
                if zone_key not in f:
                    raise RuntimeError(f"Zone {load_zone} not found in {dataset_path}")

                zone_group = f[zone_key]
                for traj_key in zone_group:
                    traj_data = {}
                    traj_group = zone_group[traj_key]

                    # Extract trajectory data
                    for key in traj_group:
                        dataset = traj_group[key]
                        if dataset.shape == ():  # scalar dataset
                            traj_data[key] = dataset[()]
                        else:
                            traj_data[key] = dataset[:]

                    # Extract metadata from attributes
                    for attr_key in traj_group.attrs:
                        traj_data[attr_key] = traj_group.attrs[attr_key]

                    trajectories.append(traj_data)

        except Exception as e:
            print(f"[ERROR] Failed to load from path: {dataset_path}")
            raise RuntimeError(f"Failed to load dataset from {dataset_path}: {e}")

        # Convert lists back to NumPy arrays and normalize data
        states, traj_lens, returns = [], [], []
        for traj in trajectories:
            traj["observations"] = np.array(traj["observations"], dtype=np.float32)
            traj["rewards"]      = np.array(traj["rewards"], dtype=np.float32)
            traj["actions"]      = np.array(traj["actions"], dtype=np.float32)

            states.append(traj["observations"])
            traj_lens.append(len(traj["observations"]))
            returns.append(sum(traj['rewards']))

        traj_lens, returns = np.array(traj_lens), np.array(returns)
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        # Print dataset statistics
        print("=" * 50)
        print(f"Dataset for Zone {load_zone} loaded successfully from {dataset_path}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print("=" * 50)

        # Sort and filter trajectories by return
        sorted_inds = np.argsort(returns)  # Lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        trajectories = [trajectories[ii] for ii in sorted_inds]

        return trajectories, state_mean, state_std



    def _augment_trajectories(
        self,
        online_envs,
        target_explore,
        online_iter,
        n,
        randomized=False
    ):
        max_ep_len = MAX_EPISODE_LEN
    
        with torch.no_grad():
            # generate init state
            target_return = [target_explore * self.reward_scale] * 1

            
            returns, lengths, trajs, metrics = vec_evaluate_episode_rtg(
                online_envs,
                self.chargers,
                self.routes,
                self.state_dim,
                self.act_dim,
                self.environment.num_cars,
                self.zone_index,
                online_iter,
                self.agg_num,
                self.model,
                self.arwt,
                max_ep_len=max_ep_len,
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="normal",
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
            )    
        # Add trajectories to replay buffer
        self.online_dataset.update_with_new_trajectories(trajs)

        
        self.aug_trajs += trajs
        self.total_transitions_sampled += np.sum(lengths)
    
        return {
            "aug_traj/return": np.mean(returns),
            "aug_traj/length": np.mean(lengths),
        }, metrics


    def pretrain(self, eval_envs, loss_fn):
        print("\n\n\n*** Pretrain ***")

        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                chargers=self.chargers,
                routes=self.routes,
                num_cars=self.environment.num_cars,
                zone_index=self.zone_index,
                episode_num=self.pretrain_iter,
                aggregation_num=self.agg_num,
                average_rewards_when_training=self.arwt,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )
        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            # in every iteration, prepare the data loader
            dataloader = create_dataloader(
                trajectories=self.offline_trajs,
                num_iters=self.variant["num_updates_per_pretrain_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )
            eval_outputs, eval_reward = self.evaluate(eval_fns)
            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            outputs.update(eval_outputs)
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=True,
            )

            self.pretrain_iter += 1

    def evaluate(self, eval_fns):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        for eval_fn in eval_fns:
            o = eval_fn(self.model)
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return_mean_gm"]
        return outputs, eval_reward
        
    def save_attn_layers(self, model, device):
        # Get the number of attention layers and their dimensions
        attn_number = len(model.transformer.h)
        attn_layer_size = model.transformer.h[0].attn.c_attn.weight.shape
        
        # Extract attention layers into a tensor
        attn_layers = torch.empty((attn_number, attn_layer_size[0], attn_layer_size[1]), dtype=torch.float32, device=self.device)
        for i in range(attn_number):
            attn_layers[i, :, :] = model.transformer.h[i].attn.c_attn.weight
        self.attn_layers = attn_layers

    def set_attn_layers(self, model, attn_layers):
        attn_number = len(model.transformer.h)
        for i in range(attn_number):
            model.transformer.h[i].attn.c_attn.weight = torch.nn.Parameter(attn_layers[i])
    
        return model
            
    def get_model_weights(self):
        return self.attn_layers

    def online_tuning(self, online_envs, eval_envs, loss_fn):
        print("\n\n\n*** Online Finetuning ***")
    
        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                chargers=self.chargers,
                routes=self.routes,
                num_cars=self.environment.num_cars,
                zone_index=self.zone_index,
                episode_num=self.online_iter,
                aggregation_num=self.agg_num,
                average_rewards_when_training=self.arwt,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]
        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )
    
        full_metrics = []

        
        while self.online_iter < self.variant["max_online_iters"]:
            outputs = {}   
            augment_outputs, metrics = self._augment_trajectories(
                online_envs,
                self.variant["online_rtg"],
                self.online_iter,
                n=self.variant["num_online_rollouts"],
                
            )
            full_metrics.extend(metrics)
            outputs.update(augment_outputs)
            
            # Call evaluate() only every 5 iterations
            if self.online_iter % 5 == 0:
                directory = f"{self.metrics_base_path}/train/metrics"
                os.makedirs(directory, exist_ok=True)
                evaluate(self.ev_info, full_metrics, self.seed, self.date, self.verbose, 'save', self.variant["max_online_iters"], directory, True, True)
                full_metrics = []
    
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant["eval_interval"] == 0 or is_last_iter:
                evaluation = True
            else:
                evaluation = False
    
            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=self.online_dataloader,
            )
            outputs.update(train_outputs)
    
            if evaluation:
                eval_outputs, eval_reward = self.evaluate(eval_fns)
                outputs.update(eval_outputs)
    
            outputs["time/total"] = time.time() - self.start_time
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter + self.online_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=False
            )
    
            if is_last_iter:
                self.save_attn_layers(self.model, self.device)
                
            self.online_iter += 1 


    def __call__(self):

        utils.set_seed_everywhere(self.seed)

        def loss_fn(
            a_hat_dist,
            a,
            attention_mask,
            entropy_reg,
        ):
            # a_hat is a SquashedNormal Distribution
            log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

            entropy = a_hat_dist.entropy().mean()
            loss = -(log_likelihood + entropy_reg * entropy)

            return (
                loss,
                -log_likelihood,
                entropy,
            )

        def get_env_builder(seed, env_name, target_goal=None):
            def make_env_fn():
                env = self.environment
                return env

            return make_env_fn

        print("\n\nMaking Eval Env.....")
        env_name = 'merl'
        target_goal = None
        eval_envs = self.environment

        self.start_time = time.time()
        if self.agg_num < 1 and self.variant["max_pretrain_iters"]:
            self.pretrain(eval_envs, loss_fn)

        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            online_envs = self.environment
            self.online_tuning(online_envs, eval_envs, loss_fn)
            

def train_odt(ev_info, metrics_base_path, experiment_number, chargers, environment, routes, date, action_dim, global_weights, aggregation_num, zone_index, seed, main_seed, device, agent_by_zone, variant, args, fixed_attributes=None, verbose=False, display_training_times=False, 
              dtype=torch.float32, save_offline_data=False, train_model=True, old_buffers=None):

    num_episodes = variant['nn_hyperparameters']['num_episodes']
    arwt = variant['nn_hyperparameters']['average_rewards_when_training']
    num_aggs = variant['federated_learning_settings']['aggregation_count']
    variant = variant['odt_hyperparameters']
    variant["max_online_iters"] = num_episodes
    utils.set_seed_everywhere(main_seed)
    
    print(f'episode calc check: {variant["max_online_iters"] }')
    print(f"Evaluation: {variant['evaluation']}")
    
    experiment = Experiment(variant, environment, chargers, routes, environment.state_dim, action_dim, device, main_seed, experiment_number, aggregation_num, zone_index, old_buffers, metrics_base_path, ev_info, date, verbose, num_episodes, arwt)

    print("=" * 50)
    experiment()

    clean_buffers = []
    for traj in experiment.online_dataset.trajectories:
        clean = {}
        for k, v in traj.items():
            if isinstance(v, torch.Tensor):
                clean[k] = v.detach().cpu().numpy()
            else:
                clean[k] = v
        clean_buffers.append(clean)

    return (
        experiment.get_model_weights().detach().cpu(),
        [], [], 
        experiment.metrics,
        clean_buffers
    )

def load_global_weights(save_global_path):
    """
    Function to load global weights from the specified path and convert to a dictionary if necessary.

    Parameters:
    - save_global_path (str): Path to the saved global weights.

    Returns:
    - global_weights (dict): Loaded global weights, converted to a dictionary if originally a list.
    """
    weights_path = f'{save_global_path}/global_weights.pth'
    if Path(weights_path).exists():
        global_weights = torch.load(weights_path)

        # Check if global_weights is a list and convert to dictionary if needed
        if isinstance(global_weights, list):
            # Example conversion logic (adjust as per your needs)
            global_weights_dict = {}
            for i, weights_dict in enumerate(global_weights):
                if isinstance(weights_dict, dict):
                    for key, value in weights_dict.items():
                        # Create a unique key name if necessary
                        global_weights_dict[f"zone_{i}_{key}"] = value
                else:
                    raise ValueError("Expected a list of dictionaries, but found a different structure.")
            global_weights = global_weights_dict

            #print(f'Global_weights shape: {len(global_weights)}')
        return global_weights
    else:
        raise FileNotFoundError(f"No global weights found at {weights_path}")