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
            mode="normal",
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            use_mean=use_mean,
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
    device="cuda",
    mode="normal",
    use_mean=False,
):
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
    #Maybe move outside of this file


    while not sim_done:
        timestep_counter = environment.init_routing()
        start_time_step = time.time()

        cur_len = trajectories[0]['cur_len']

        # Phase 1: collect states from all cars (sequential — env interaction)
        for car in range(num_cars):
            state = environment.reset_agent(car, False)
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
            torch.tensor(target_return, device=device).reshape(1, 1).expand(num_cars, 1),
            torch.tensor(timestep_counter, device=device, dtype=torch.long).reshape(1, 1).expand(num_cars, 1),
            num_envs=num_cars,
        )
        all_actions_tanh = action_dist.mean[:, -1, :]  # (num_cars, act_dim)

        # Phase 3: dispatch actions to all cars (sequential — env interaction)
        for car in range(num_cars):
            action_tanh = all_actions_tanh[car]
            if cur_len < max_traj_len:
                trajectories[car]['actions'][cur_len] = action_tanh.detach()
            environment.generate_paths((action_tanh + 1) / 2, None, car)
      
        try:
            sim_done, timestep_reward, arrived_at_final = environment.simulate_routes()
        except Exception as e:
            if "NEGATIVE BATTERY" in str(e):
                print(f"[WARN] Negative battery during evaluation — treating as terminal with penalty.", flush=True)
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