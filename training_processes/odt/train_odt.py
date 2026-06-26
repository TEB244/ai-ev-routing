from torch.utils.tensorboard import SummaryWriter

import pickle
import time
import torch
import numpy as np
import re
import os
import glob
import h5py
import getpass
import socket

from decision_makers.agent_odt import DecisionTransformer

from .odt_helpers import utils
from .odt_helpers.replay_buffer import ReplayBuffer
from .odt_helpers.data import create_dataloader, TransformSamplingSubTraj
from .odt_helpers.evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from .odt_helpers.trainer import SequenceTrainer
from .odt_helpers.logger import Logger
from .odt_helpers.online_data import PersistentOnlineDataset, create_online_dataloader

class NullCarbonTracker:
    def epoch_start(self): pass
    def epoch_end(self): pass
    def stop(self): pass


class Experiment:
    def __init__(self, params):
        self.queue               = params['queue']
        self.ev_info             = params['ev_info']
        self.metrics_base_path   = params['metrics_base_path']
        self.experiment_number   = params['experiment_number']
        self.chargers            = params['chargers']
        self.environment         = params['environment']
        self.routes              = params['routes']
        self.date                = params['date']
        self.action_dim          = params['action_dim']
        self.global_weights      = params['global_weights']
        self.aggregation_num     = params['aggregation_num']
        self.zone_index          = params['zone_index']
        self.seed                = params['seed']
        self.main_seed           = params['main_seed']
        self.device              = params['device']
        self.config              = params['config']
        self.args                = params['args']
        self.verbose             = params['verbose']
        self.old_buffers         = params['old_buffers']
        self.Max_episode_len     = params['Max_episode_len']

        self.evaluation = bool(getattr(self.args, "eval", False))
        self.reward_scale = 1
        self.action_range = [-1, 1]
        self.arwt = self.config['nn_hyperparameters']['average_rewards_when_training']
        self.eps_per_save = int(self.config['nn_hyperparameters'].get('eps_per_save', 1))

        # Retrieve DRAC username and cluster
        drac_username = os.environ.get("DRAC_USERNAME") or getpass.getuser()
        # cluster_str = socket.gethostname()
        # if "rorqual" in cluster_str:
        #     data_dir = f"/home/{drac_username}/links/scratch"
        # else:
        #     data_dir = f"/home/{drac_username}/scratch"
        data_dir = f"/home/{drac_username}/links/scratch"
        scratch_base = os.path.expanduser(os.path.join(data_dir, "saved_networks"))
        if os.path.isdir(os.path.expanduser("~/scratch")):
            self.base_dir = os.path.join(scratch_base, f"Exp_{self.experiment_number}")
        else:
            self.base_dir = f"saved_networks/Exp_{self.experiment_number}"

        # Setup logger
        if self.evaluation:
            self.base_dir = os.path.join(self.base_dir, "eval")
        os.makedirs(self.base_dir, exist_ok=True)
        stats_dir = os.path.join(self.base_dir, "StateStats")
        os.makedirs(stats_dir, exist_ok=True)
        self.stats_path = os.path.join(stats_dir, f"state_stats_zone_{self.zone_index}.pkl")
        
        self.logger = Logger(self.base_dir, self.experiment_number,
                             self.aggregation_num, self.zone_index)
    
    def init_agent(self):
        self.start_time = time.time()
        self.odt_config = self.config['odt_hyperparameters']
        
        #Instantiate agent
        self.ODTAgent = DecisionTransformer(
            state_dim=self.environment.state_dim,
            act_dim=self.action_dim,
            action_range=self.action_range,
            max_ep_len=self.Max_episode_len,
            n_positions=1024,
            stochastic_policy=True,
            target_entropy=-self.action_dim,
            odt_config=self.odt_config
        ).to(device=self.device)
        
        self.ODTAgent.set_optimizer_scheduler(self.odt_config)

        #Loads previously trained model weights from different zone if evaluation
        if self.evaluation and self.aggregation_num == 0:
            eval_cfg = self.config.get('eval_config', {})
            inference_only = bool(getattr(self.args, 'eval', False)
                                  and eval_cfg.get('inference_only', False))
            if inference_only:
                # Forward-only inference (10xxx energy batch). Load the *converged*
                # pretrained model for THIS zone from a (possibly different) source
                # experiment. Training saves a checkpoint per aggregation/zone as
                # 'Agg:<n>-Zone:<z>/model.pt' (z is 0-indexed); we load the HIGHEST
                # aggregation = the fully-trained model. _load_weights() appends
                # '/model.pt', so we pass the containing directory.
                #
                # (The legacy branch below loaded Agg:0 with a 1-indexed Zone+1 and
                # passed a file path as the prefix -- all three were wrong, so eval
                # silently loaded nothing.)
                src_exp = eval_cfg.get('weights_from_exp') or self.experiment_number
                zone = self.zone_index  # 0-indexed, matches the Agg:n-Zone:z dir names
                search_bases = [
                    f"saved_networks/Exp_{src_exp}",
                    os.path.expanduser(f"~/links/scratch/saved_networks/Exp_{src_exp}"),
                    os.path.expanduser(f"~/scratch/saved_networks/Exp_{src_exp}"),
                ]
                hits = []
                for base in search_bases:
                    hits = glob.glob(
                        os.path.join(base, "**", f"Agg:*-Zone:{zone}", "model.pt"),
                        recursive=True)
                    if hits:
                        break
                if not hits:
                    raise FileNotFoundError(
                        f"ODT inference: no 'Agg:*-Zone:{zone}/model.pt' for source "
                        f"Exp_{src_exp}. Searched (recursively): {search_bases}")

                def _agg_num(path):
                    m = re.search(r"Agg:(\d+)-Zone:", path)
                    return int(m.group(1)) if m else -1

                model_file = max(hits, key=_agg_num)
                model_dir = os.path.dirname(model_file)
                print(f"[ODT inference] zone {zone}: loading {model_file} "
                      f"(agg {_agg_num(model_file)}) from source Exp_{src_exp}")
                self.ODTAgent._load_weights(path_prefix=model_dir, load_optimizer=False)
            else:
                num_zones = self.odt_config.get('num_zones', 4)
                next_zone = (self.zone_index + 1) % num_zones
                model_dir = f"../exp/Exp_{self.experiment_number}/Agg:0-Zone:{next_zone+1}"
                model_file = model_dir + "/model.pt"

                self.ODTAgent._load_weights(path_prefix=model_file, load_optimizer=True)
            
        # Load previous aggregation weights if on new aggregation
        if self.aggregation_num > 0:
            prev_log = re.sub(r"Agg:\d+", f"Agg:{self.aggregation_num-1}", self.logger.log_path)
            self.ODTAgent._load_weights(path_prefix=prev_log, load_optimizer=False)

            global_path = f"saved_networks/Exp_{self.experiment_number}"
            self.ODTAgent.load_attn_layers(global_path)
        
    def train_offline(self):
        total_transitions_sampled = 0
        if self.evaluation:
            num_zones = self.odt_config.get("num_zones", 4)
            load_zone = (self.zone_index + 1) % num_zones
        else:
            load_zone = self.zone_index
    
        # Locate the dataset file
        data_dir = self.odt_config['offline_dataset_path']
        available = len(glob.glob(os.path.join(data_dir, "data_zone_*.h5")))
        if available > 0:
            load_zone = load_zone % available
        dataset_path = os.path.join(data_dir, f"data_zone_{load_zone}.h5")
    
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"No .h5 files found for zone {load_zone} in {data_dir}. Exiting...")
            exit(1)
    
        max_trajs = self.odt_config.get('max_offline_trajectories', None)

        trajectories = []
        try:
            with h5py.File(dataset_path, 'r') as f:
                zone_key = f"zone_{load_zone}"
                if zone_key not in f:
                    raise RuntimeError(f"Zone {load_zone} not found in {dataset_path}")

                zone_group = f[zone_key]
                all_keys = sorted(zone_group.keys(), key=lambda k: int(k.split('_')[1]))
                if max_trajs is not None:
                    all_keys = all_keys[-max_trajs:]
                    print(f"[INFO] Loading last {len(all_keys)} trajectories from {len(zone_group)} total")

                for traj_key in all_keys:
                    traj_data = {}
                    traj_group = zone_group[traj_key]

                    for key in traj_group:
                        dataset = traj_group[key]
                        if dataset.shape == ():
                            traj_data[key] = dataset[()]
                        else:
                            traj_data[key] = dataset[:]

                    for attr_key in traj_group.attrs:
                        traj_data[attr_key] = traj_group.attrs[attr_key]

                    trajectories.append(traj_data)

        except Exception as e:
            print(f"[ERROR] Failed to load from path: {dataset_path}")
            raise RuntimeError(f"Failed to load dataset from {dataset_path}: {e}")
    
        # Convert lists back to NumPy arrays and normalize data
        states, traj_lens, returns = [], [], []
        n_dropped = 0
        for traj in trajectories:
            traj["observations"] = np.array(traj["observations"], dtype=np.float32)
            traj["rewards"]      = np.array(traj["rewards"], dtype=np.float32)
            traj["actions"]      = np.array(traj["actions"], dtype=np.float32)

            if (np.any(~np.isfinite(traj["observations"])) or
                    np.any(~np.isfinite(traj["rewards"])) or
                    np.any(~np.isfinite(traj["actions"]))):
                n_dropped += 1
                continue

            states.append(traj["observations"])
            traj_lens.append(len(traj["observations"]))
            returns.append(sum(traj['rewards']))

        if n_dropped:
            print(f"[WARN] Dropped {n_dropped} trajectories containing NaN/inf values")

        traj_lens, returns = np.array(traj_lens), np.array(returns)
        states = np.concatenate(states, axis=0)

        bad_rows = ~np.all(np.isfinite(states), axis=1)
        if bad_rows.any():
            print(f"[WARN] Dropping {bad_rows.sum()} NaN/inf timesteps from offline dataset before normalisation")
            states = states[~bad_rows]

        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        print(f"[INFO] state_mean min/max: {state_mean.min():.4f}/{state_mean.max():.4f}  "
              f"state_std min/max: {state_std.min():.6f}/{state_std.max():.4f}")
        num_timesteps = sum(traj_lens)
    
        print("=" * 50)
        print(f"Dataset for Zone {load_zone} loaded successfully from {dataset_path}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps")
        print(f"Return stats: mean={np.mean(returns):.2f}, std={np.std(returns):.2f}, "
              f"min={np.min(returns):.2f}, max={np.max(returns):.2f}, "
              f"p5={np.percentile(returns, 5):.2f}, p25={np.percentile(returns, 25):.2f}, "
              f"p75={np.percentile(returns, 75):.2f}, p95={np.percentile(returns, 95):.2f}")
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
        with open(self.stats_path, 'wb') as f:
            pickle.dump({'state_mean': state_mean, 'state_std': state_std}, f)

        offline_iter = 0
        print("\n\n\n*** Offline Training ***")
        if getattr(self.args, 'server', 'DRAC') == 'DRAC':
            self.tracker = NullCarbonTracker()
        else:
            from carbontracker.tracker import CarbonTracker
            self.tracker = CarbonTracker(epochs=(self.odt_config["max_pretrain_iters"] + self.odt_config["max_online_iters"]), epochs_before_pred=0, monitor_epochs=-1, update_interval=1, verbose=0, ignore_errors=True)
        eval_fns = [
            create_vec_eval_episodes_fn(
                queue=self.queue,
                vec_env=self.environment,
                eval_rtg=self.odt_config["eval_rtg"],
                state_dim=self.environment.state_dim,
                act_dim=self.action_dim,
                state_mean=state_mean,
                state_std=state_std,
                device=self.device,
                chargers=self.chargers,
                routes=self.routes,
                num_cars=self.environment.num_cars,
                zone_index=self.zone_index,
                episode_num=offline_iter,
                aggregation_num=self.aggregation_num,
                average_rewards_when_training=self.arwt,
                metrics_path=self.metrics_base_path,
                use_mean=True,
                reward_scale=self.reward_scale,
                start_sequence=self.config['environment_settings'].get('start_sequence', 'static'),
                seed=self.seed,
                forward_mode=self.odt_config.get('forward_mode', 'batched'),
            )
        ]

        trainer = SequenceTrainer(
            model=self.ODTAgent,
            optimizer=self.ODTAgent.optimizer,
            log_temperature_optimizer=self.ODTAgent.log_temperature_optimizer,
            scheduler=self.ODTAgent.scheduler,
            device=self.device,
        )

        writer = (
            SummaryWriter(self.logger.log_path)
        )
        self.environment.init_sim(self.aggregation_num)
        while offline_iter < self.odt_config["max_pretrain_iters"]:
            self.tracker.epoch_start() # Start tracking carbon emissions
            dataloader = create_dataloader(
                trajectories=trajectories,
                num_iters=self.odt_config["num_updates_per_pretrain_iter"],
                batch_size=self.odt_config["batch_size"],
                max_len=self.odt_config["K"],
                state_dim=self.environment.state_dim,
                act_dim=self.action_dim,
                state_mean=state_mean,
                state_std=state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
                is_offline=True,
            )

            train_outputs = trainer.train_iteration(
                loss_fn=self.loss_fn,
                dataloader=dataloader,
            )
            eval_outputs, _ = utils.evaluateODT(eval_fns, self.ODTAgent)
            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            outputs.update(eval_outputs)
            self.logger.log_metrics(
                outputs,
                iter_num=offline_iter,
                total_transitions_sampled=total_transitions_sampled,
                writer=writer,
            )
            is_last_offline = offline_iter == self.odt_config["max_pretrain_iters"] - 1
            if (offline_iter + 1) % self.eps_per_save == 0 or is_last_offline:
                self.ODTAgent._save_weights(self.logger.log_path, is_offline_model=is_last_offline)

            offline_iter += 1
            self.tracker.epoch_end()
            try:
                # kWh used in each finished epoch; take the last one
                epoch_kwh = float(self.tracker.tracker.total_energy_per_epoch()[-1])
                # Average carbon intensity during this run (gCO2/kWh)
                avg_ci = self.tracker.intensity_updater.average_carbon_intensity()
                episode_co2_g = float(epoch_kwh * avg_ci.carbon_intensity)

                self.queue.put({
                    'tag': 'sustainability_episode',
                    'kwh': epoch_kwh,              # kWh for this episode
                    'co2': episode_co2_g,          # grams CO2e for this episode
                    'episode': offline_iter,
                    'zone_index': self.zone_index,
                    'aggregation_step': self.aggregation_num
                })
            except Exception as e:
                # If Carbontracker wasn’t able to read (e.g., permissions/NVML), push a minimal record
                self.queue.put({
                    'tag': 'sustainability_episode',
                    'kwh': None,
                    'co2': None,
                    'error': f'carbontracker_read_failed: {e}',
                    'episode': offline_iter,
                    'zone_index': self.zone_index,
                    'aggregation_step': self.aggregation_num
                })
        
        return trajectories, state_mean, state_std
        
    def train_online(self, trajectories, state_mean, state_std):     
        print("\n\n\n*** Online Training ***")

        online_iter = 0
        total_transitions_sampled = 0
        online_rng = np.random.default_rng(self.seed)

        #Create replay buffer from trajectories:
        replay_buffer = ReplayBuffer(self.odt_config['replay_size'], trajectories)

        #Tracker for consecutive aggregations
        if self.aggregation_num > 0:
            self.odt_config["max_offline_iters"] = 0 #For logging purposes
            if getattr(self.args, 'server', 'DRAC') == 'DRAC':
                self.tracker = NullCarbonTracker()
            else:
                from carbontracker.tracker import CarbonTracker
                self.tracker = CarbonTracker(epochs=self.odt_config["max_online_iters"], epochs_before_pred=0, monitor_epochs=-1, update_interval=1, verbose=0, ignore_errors=True)
        else:
            self.odt_config["max_offline_iters"] = self.odt_config["max_pretrain_iters"]
        
        #Builds persistent online dataset/replay buffer only updated with online experiences
        transform = TransformSamplingSubTraj(
            max_len=self.odt_config['K'],
            state_dim=self.environment.state_dim,
            act_dim=self.action_dim,
            state_mean=state_mean,
            state_std=state_std,
            reward_scale=self.reward_scale,
            action_range=self.action_range
        )
        online_dataset = PersistentOnlineDataset(
            initial_trajectories=replay_buffer.trajectories,
            sample_size=self.odt_config['batch_size'] * self.odt_config['num_updates_per_online_iter'],
            transform=transform
        )
        trainer = SequenceTrainer(
            model=self.ODTAgent,
            optimizer=self.ODTAgent.optimizer,
            log_temperature_optimizer=self.ODTAgent.log_temperature_optimizer,
            scheduler=self.ODTAgent.scheduler,
            device=self.device,
        )
        eval_fns = [
            create_vec_eval_episodes_fn(
                queue=self.queue,
                vec_env=self.environment,
                eval_rtg=self.odt_config["eval_rtg"],
                state_dim=self.environment.state_dim,
                act_dim=self.action_dim,
                state_mean=state_mean,
                state_std=state_std,
                device=self.device,
                chargers=self.chargers,
                routes=self.routes,
                num_cars=self.environment.num_cars,
                zone_index=self.zone_index,
                episode_num=online_iter,
                aggregation_num=self.aggregation_num,
                average_rewards_when_training=self.arwt,
                metrics_path=self.metrics_base_path,
                use_mean=True,
                reward_scale=self.reward_scale,
                start_sequence=self.config['environment_settings'].get('start_sequence', 'static'),
                seed=self.seed,
                forward_mode=self.odt_config.get('forward_mode', 'batched'),
            )
        ]
        writer = (
            SummaryWriter(self.logger.log_path)
        )

        self.environment.init_sim(self.aggregation_num)
        while online_iter < self.odt_config["max_online_iters"]:
            self.tracker.epoch_start()
            online_dataloader = create_online_dataloader(
                online_dataset,
                batch_size=self.odt_config['batch_size']
            )
            outputs = {} 
            with torch.no_grad(): 
                target_return = [self.odt_config["online_rtg"] * self.reward_scale]
                returns, lengths, trajs = vec_evaluate_episode_rtg(
                    self.queue,
                    self.environment,
                    self.chargers,
                    self.routes,
                    self.environment.state_dim,
                    self.action_dim,
                    self.environment.num_cars,
                    self.zone_index,
                    online_iter,
                    self.aggregation_num,
                    self.ODTAgent,
                    self.arwt,
                    self.metrics_base_path,
                    max_ep_len=self.Max_episode_len,
                    reward_scale=self.reward_scale,
                    target_return=target_return,
                    mode=self.odt_config.get('forward_mode', 'batched'),
                    state_mean=state_mean,
                    state_std=state_std,
                    device=self.device,
                    start_sequence=self.config['environment_settings'].get('start_sequence', 'static'),
                    seed=online_rng,
                )
            online_dataset.update_with_new_trajectories(trajs)
            total_transitions_sampled += int(np.sum(lengths))
            
            augment_outputs = {
                "aug_traj/return": float(np.mean(returns)),
                "aug_traj/length": float(np.mean(lengths)),
            }
            
            outputs.update(augment_outputs)  

            is_last_iter = online_iter == self.odt_config["max_online_iters"] - 1
            if (online_iter + 1) % self.odt_config["eval_interval"] == 0 or is_last_iter:
                evalODT = True
            else:
                evalODT = False
    
            train_outputs = trainer.train_iteration(
                loss_fn=self.loss_fn,
                dataloader=online_dataloader,
            )
            outputs.update(train_outputs)
    
            if evalODT:
                eval_outputs, _ = utils.evaluateODT(eval_fns, self.ODTAgent)
                outputs.update(eval_outputs)
    
            outputs["time/total"] = time.time() - self.start_time
            self.logger.log_metrics(
                outputs,
                iter_num=online_iter,
                total_transitions_sampled=total_transitions_sampled,
                writer=writer,
            )
            if (online_iter + 1) % self.eps_per_save == 0 or is_last_iter:
                self.ODTAgent._save_weights(self.logger.log_path, is_offline_model=False)

            if is_last_iter:
                attn_layers = self.ODTAgent.get_attn_layers(self.device)
            online_iter += 1

            self.tracker.epoch_end()
            try:
                # kWh used in each finished epoch; take the last one
                epoch_kwh = float(self.tracker.tracker.total_energy_per_epoch()[-1])
                # Average carbon intensity during this run (gCO2/kWh)
                avg_ci = self.tracker.intensity_updater.average_carbon_intensity()
                episode_co2_g = float(epoch_kwh * avg_ci.carbon_intensity)

                self.queue.put({
                    'tag': 'sustainability_episode',
                    'kwh': epoch_kwh,              # kWh for this episode
                    'co2': episode_co2_g,          # grams CO2e for this episode
                    'episode': online_iter + self.odt_config["max_offline_iters"],
                    'zone_index': self.zone_index,
                    'aggregation_step': self.aggregation_num
                })
            except Exception as e:
                # If Carbontracker wasn’t able to read (e.g., permissions/NVML), push a minimal record
                self.queue.put({
                    'tag': 'sustainability_episode',
                    'kwh': None,
                    'co2': None,
                    'error': f'carbontracker_read_failed: {e}',
                    'episode': online_iter + self.odt_config["max_offline_iters"],
                    'zone_index': self.zone_index,
                    'aggregation_step': self.aggregation_num
                })

        self.tracker.stop() # Stop tracking carbon emissions
        return attn_layers.detach().cpu(), online_dataset

    def loss_fn(self, a_hat_dist, a, attention_mask, entropy_reg):
        # a_hat is a SquashedNormal Distribution
        log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()
        entropy = a_hat_dist.entropy().mean()
        loss = -(log_likelihood + entropy_reg * entropy)
        return (loss, -log_likelihood, entropy)

def train_odt(
    queue,#Not used
    metrics_base_path,
    ev_info,
    experiment_number,
    chargers,
    environment,
    routes,
    date,
    action_dim,
    global_weights,
    aggregation_num,
    zone_index,
    seed,
    main_seed,
    device,
    agent_by_zone,
    config,
    args,
    fixed_attributes=None,
    verbose=False,
    display_training_times=False,
    dtype=torch.float32,
    save_offline_data=False,
    train_model=True,
    old_buffers=None
):
    utils.set_seed_everywhere(int(seed))
    config['odt_hyperparameters']['max_online_iters'] = config['nn_hyperparameters']['num_episodes']
    params = {
        "queue":                 queue,
        "ev_info":               ev_info,
        "metrics_base_path":     metrics_base_path,
        "experiment_number":     experiment_number,
        "chargers":              chargers,
        "environment":           environment,
        "routes":                routes,
        "date":                  date,
        "action_dim":            action_dim,
        "global_weights":        global_weights,
        "aggregation_num":       aggregation_num,
        "zone_index":            zone_index,
        "seed":                  seed,
        "main_seed":             main_seed,
        "device":                device,
        "config":                config,
        "args":                  args,
        "verbose":               verbose,
        "old_buffers":           old_buffers,
        "Max_episode_len":       1000
    }

    experiment = Experiment(params)
    
    
    #Initialize agent
    # experiment.init_agent()
    try:
        experiment.init_agent()
    except RuntimeError as e:
        if "CUDA-capable device(s) is/are busy" in str(e):
            print(f"CUDA-capable device(s) is busy but experiment continues")
            pass  # silently skip, experiment continues
        else:
            raise
    
    #On first aggregation, load dataset and train offline
    if aggregation_num == 0 and experiment.odt_config["max_pretrain_iters"] > 0:
        trajectories, state_mean, state_std = experiment.train_offline()
        for traj in trajectories:
            traj['actions'] = 2 * traj['actions'] - 1
    elif aggregation_num == 0:
        # No pretrain requested — skip offline data loading, use unit normalization
        trajectories = []
        state_mean = np.zeros(experiment.environment.state_dim, dtype=np.float32)
        state_std = np.ones(experiment.environment.state_dim, dtype=np.float32)
        with open(experiment.stats_path, 'wb') as f:
            pickle.dump({'state_mean': state_mean, 'state_std': state_std}, f)
        if getattr(args, 'server', 'DRAC') == 'DRAC':
            experiment.tracker = NullCarbonTracker()
        else:
            from carbontracker.tracker import CarbonTracker
            experiment.tracker = CarbonTracker(epochs=config['odt_hyperparameters']['max_online_iters'], epochs_before_pred=0, monitor_epochs=-1, update_interval=1, verbose=0, ignore_errors=True)
    #Otherwise, get previous agg trajectories and dataset stats
    else:
        trajectories = old_buffers
        with open(experiment.stats_path, 'rb') as f:
            stats = pickle.load(f)
            state_mean, state_std = stats['state_mean'], stats['state_std']
            
    #Train online
    attn_layers, online_dataset = experiment.train_online(trajectories, state_mean, state_std)
    
    #Clean buffers(maintains trajectories between aggregations) before passing back
    buffers = []
    for traj in online_dataset.trajectories:
        clean = {}
        for k, v in traj.items():
            if isinstance(v, torch.Tensor):
                clean[k] = v.detach().cpu().numpy()
            else:
                clean[k] = v
        buffers.append(clean)
        
    return attn_layers, [], [], buffers