"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import time

from training_processes.writer_proccess import printer_queue

MAX_EPISODE_LEN = 1000


def create_vec_eval_episodes_fn(
    queue,
    vec_env,
    eval_rtg,
    state_dim,
    act_dim,
    state_mean,
    state_std,
    device,
    chargers,
    routes,
    num_cars,
    zone_index,
    episode_num,
    aggregation_num,
    average_rewards_when_training,
    metrics_path,
    use_mean=False,
    reward_scale=0.001,
    start_sequence='static',
    seed=None,
    forward_mode='batched',
):
    def eval_episodes_fn(model):
        target_return = [eval_rtg * reward_scale] * 1
        returns, lengths, _ = vec_evaluate_episode_rtg(
            queue,
            vec_env,
            chargers,
            routes,
            state_dim,
            act_dim,
            num_cars,
            zone_index,
            episode_num,
            aggregation_num,
            model,
            average_rewards_when_training,
            metrics_path,
            max_ep_len=MAX_EPISODE_LEN,
            reward_scale=reward_scale,
            target_return=target_return,
            mode=forward_mode,
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            use_mean=use_mean,
            start_sequence=start_sequence,
            seed=seed,
        )
        suffix = "_gm" if use_mean else ""
        return {
            f"evaluation/return_mean{suffix}": np.mean(returns),
            f"evaluation/return_std{suffix}": np.std(returns),
            f"evaluation/length_mean{suffix}": np.mean(lengths),
            f"evaluation/length_std{suffix}": np.std(lengths),
        }

    return eval_episodes_fn


@torch.no_grad()
def vec_evaluate_episode_rtg(
    queue,
    environment,
    chargers,
    routes,
    state_dim,
    act_dim,
    num_cars,
    zone_index,
    episode_num,
    aggregation_num,
    model,
    average_rewards_when_training,
    metrics_path,
    target_return: list,
    max_ep_len=10,
    reward_scale=0.001,
    state_mean=0.0,
    state_std=1.0,
    start_sequence='static',
    seed=None,
    device="cuda",
    mode="normal",
    use_mean=False,
):
    eval_rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    # Move state_mean and state_std to the device once
    state_mean = torch.tensor(state_mean, device=device) if isinstance(state_mean, np.ndarray) else state_mean
    state_std = torch.tensor(state_std, device=device) if isinstance(state_std, np.ndarray) else state_std
    
    assert len(target_return) == 1
    unique_chargers = np.unique(np.array(list(map(tuple, chargers.reshape(-1, 3))), dtype=[('id', int), ('lat', float), ('lon', float)]))
    environment.reset_episode(chargers, routes, unique_chargers)
    model.eval()
    model.to(device=device)
    
    # Pre-allocate memory for trajectories for all cars with a fixed max length
    max_traj_len = max_ep_len
    trajectories = [{
        'observations': torch.zeros((max_traj_len, state_dim), device=device, dtype=torch.float32),
        'actions': torch.zeros((max_traj_len, act_dim), device=device, dtype=torch.float32),
        'rewards': torch.zeros(max_traj_len, device=device, dtype=torch.float32),
        'terminals': torch.zeros(max_traj_len, device=device, dtype=torch.bool),
        'car_num': car,
        'cur_len': 0  # Track the current length of each trajectory
    } for car in range(num_cars)]

    sim_done = False
    timestep_counter = 0
    episode_rewards = []
    dones = []

    # Per-car running RTG buffer: pre-allocated like obs/action/reward buffers.
    # Column t holds RTG_t = target_return - sum(rewards[0..t-1]).
    rtg_buffer = torch.zeros((num_cars, max_traj_len + 1), device=device, dtype=torch.float32)
    rtg_buffer[:, 0] = target_return[0]

    while not sim_done:
        timestep_counter = environment.init_routing()
        start_time_step = time.time()

        cur_len = trajectories[0]['cur_len']

        if start_sequence == 'random':
            starting_car = int(eval_rng.integers(0, num_cars))
        else:
            starting_car = 0

        if mode == 'sequential':
            # One car at a time: reset_agent → forward pass → generate_paths
            # Mirrors DQN exactly — traffic accumulates between cars within a sim-step.
            for i in range(num_cars):
                car = (starting_car + i) % num_cars
                state = environment.reset_agent(car, False)
                if cur_len < max_traj_len:
                    trajectories[car]['observations'][cur_len] = torch.from_numpy(state).to(device=device, dtype=torch.float32)
                obs = trajectories[car]['observations'][:cur_len + 1].unsqueeze(0)
                acts = trajectories[car]['actions'][:cur_len].unsqueeze(0)
                rews = trajectories[car]['rewards'][:cur_len].unsqueeze(0)
                _, action_dist, _ = model.get_predictions(
                    (obs - state_mean) / state_std,
                    acts,
                    rews,
                    rtg_buffer[car, :cur_len + 1].unsqueeze(0),
                    torch.tensor(timestep_counter, device=device, dtype=torch.long).reshape(1, 1),
                    num_envs=1,
                )
                action_tanh = action_dist.mean[0, -1, :]
                if cur_len < max_traj_len:
                    trajectories[car]['actions'][cur_len] = action_tanh.detach()
                environment.generate_paths((action_tanh + 1) / 2, None, car)
        else:
            # Phase 1: collect states from all cars (sequential — env interaction)
            # Cache self.agent per car — generate_paths reads self.agent and reset_agent
            # overwrites it, so without caching every car in Phase 3 would use the last car's data.
            saved_agents = {}
            for i in range(num_cars):
                car = (starting_car + i) % num_cars
                state = environment.reset_agent(car, False)
                saved_agents[car] = environment.agent
                if cur_len < max_traj_len:
                    trajectories[car]['observations'][cur_len] = torch.from_numpy(state).to(device=device, dtype=torch.float32)

            # Phase 2: one batched forward pass for all cars
            batched_obs = torch.stack([t['observations'][:cur_len + 1] for t in trajectories])
            batched_actions = torch.stack([t['actions'][:cur_len] for t in trajectories])
            batched_rewards = torch.stack([t['rewards'][:cur_len] for t in trajectories])
            _, action_dist, _ = model.get_predictions(
                (batched_obs - state_mean) / state_std,
                batched_actions,
                batched_rewards,
                rtg_buffer[:, :cur_len + 1],
                torch.tensor(timestep_counter, device=device, dtype=torch.long).reshape(1, 1).expand(num_cars, 1),
                num_envs=num_cars,
            )
            all_actions_tanh = action_dist.mean[:, -1, :]  # (num_cars, act_dim)

            # Phase 3: dispatch actions to all cars (sequential — env interaction)
            for i in range(num_cars):
                car = (starting_car + i) % num_cars
                environment.agent = saved_agents[car]  # restore before generate_paths reads self.agent
                action_tanh = all_actions_tanh[car]
                if cur_len < max_traj_len:
                    trajectories[car]['actions'][cur_len] = action_tanh.detach()
                environment.generate_paths((action_tanh + 1) / 2, None, car)
      
        try:
            sim_done, timestep_reward, arrived_at_final = environment.simulate_routes()
        except Exception as e:
            if "NEGATIVE BATTERY" in str(e):
                import traceback, os
                log_path = os.path.join(os.path.dirname(metrics_path), "negative_battery.log")
                with open(log_path, "a") as f:
                    f.write(f"\n=== episode={episode_num} agg={aggregation_num} zone={zone_index} sim_step={timestep_counter} ===\n")
                    f.write(f"starting_charge: {environment.info['starting_charge']}\n")
                    f.write(f"paths: {environment.paths}\n")
                    f.write(f"charges_needed (current):\n{environment.charges_needed}\n")
                    f.write(traceback.format_exc())
                print(f"[WARN] Negative battery — logged to {log_path}", flush=True)
                sim_done = True
                timestep_reward = np.full(num_cars, -100.0)
                arrived_at_final = np.zeros(num_cars, dtype=bool)
            else:
                raise
        
        dones.extend(arrived_at_final.tolist())
        if timestep_counter == 0:
            episode_rewards = np.expand_dims(timestep_reward,axis=0)
        else:
            episode_rewards = np.vstack((episode_rewards,timestep_reward))
        
        # Train the model only using the average of all timestep rewards
        if average_rewards_when_training: 
            avg_reward = timestep_reward.sum(axis=0).mean()
            timestep_reward_avg = [avg_reward for _ in timestep_reward]
        
        # Update rewards and terminals
        for traj in trajectories:
            if traj['cur_len'] < max_traj_len:
                if average_rewards_when_training: 
                    traj['rewards'][traj['cur_len']] = torch.tensor(timestep_reward_avg[traj['car_num']], device=device, dtype=torch.float32)
                else:
                    traj['rewards'][traj['cur_len']] = torch.tensor(timestep_reward[traj['car_num']], device=device, dtype=torch.float32)
                traj['terminals'][traj['cur_len']] = sim_done

            traj['cur_len'] += 1

        # Decrement each car's RTG by the reward just stored (cur_len was incremented above)
        cur_step = trajectories[0]['cur_len'] - 1
        rtg_buffer[:, cur_step + 1] = rtg_buffer[:, cur_step] - torch.stack(
            [traj['rewards'][cur_step] for traj in trajectories]
        )

        time_step_time = time.time() - start_time_step
            
        if timestep_counter >= environment.max_steps:
            raise Exception("MAX TIME-STEPS EXCEEDED!")

    #SIM COMPLETE  
    # Calculate the average return per car
    episode_return = np.mean(np.sum(np.vstack(episode_rewards), axis=0))
    
    station_data, agent_data, = environment.get_data()
    queue.put({
        'tag': 'csv',
        'station_data': station_data,
        'agent_data': agent_data
    })
    station_data = None
    agent_data = None

    # Truncate trajectories to actual length before returning
    trajectories = [{
        'observations': traj['observations'][:traj['cur_len']].cpu(),
        'actions': traj['actions'][:traj['cur_len']].cpu(),
        'rewards': traj['rewards'][:traj['cur_len']].cpu(),
        'terminals': traj['terminals'][:traj['cur_len']].cpu(),
        'car_num': traj['car_num']
    } for traj in trajectories]

    
    return episode_return, timestep_counter, trajectories