import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import time

from environment._pathfinding import haversine

def initialize_solver(solver_name='glpk'):
    """
    Initializes the MIP solver.

    Parameters:
        solver_name (str): The name of the solver to use (e.g., 'cbc', 'gurobi').

    Returns:
        A Pyomo solver object.
    """
    return SolverFactory(solver_name)

def get_actions(current_state_info, env_info, mpc_config, solver):
    H = mpc_config['prediction_horizon']
    alpha_d = mpc_config.get('alpha_d', 0.5)
    alpha_t = mpc_config.get('alpha_t', 0.4)
    alpha_e = mpc_config.get('alpha_e', 0.1)
    alpha_f = mpc_config.get('alpha_f', 100.0)

    model = pyo.ConcreteModel(name="EV_MPC")

    # --- Sets ---
    model.T = pyo.RangeSet(0, H - 1)
    model.T_plus_1 = pyo.RangeSet(0, H)
    model.N = pyo.Set(initialize=env_info['nodes'])
    model.C = pyo.Set(initialize=env_info['chargers'])
    model.E = pyo.Set(initialize=[(i, j) for i in model.N for j in model.N if i != j])

    # --- Variables ---
    model.a = pyo.Var(model.T, model.C, within=pyo.UnitInterval, bounds=(0.01, 0.99))
    model.x = pyo.Var(model.T, model.E, within=pyo.Binary)
    model.y = pyo.Var(model.T_plus_1, model.N, within=pyo.Binary)
    model.battery = pyo.Var(model.T_plus_1, bounds=(0.0, 100.0))
    model.path_dist = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.path_energy = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.max_traffic = pyo.Var(model.T, within=pyo.NonNegativeReals)
    
    dest_pos = env_info['destination_pos'] 
    dest_node = min(env_info['node_positions'].keys(),
                    key=lambda name: haversine(dest_pos[0], dest_pos[1], env_info['node_positions'][name][0], env_info['node_positions'][name][1]))
    dist_to_dest = {
        n: haversine(
            env_info['node_positions'][n][0], env_info['node_positions'][n][1],
            dest_pos[0], dest_pos[1]
        ) for n in model.N
    }

    # --- Objective ---
    model.objective = pyo.Objective(
        expr=sum(alpha_d * model.path_dist[t] + 
                 alpha_t * model.max_traffic[t] + 
                 alpha_e * model.path_energy[t] for t in model.T) + \
             alpha_f * sum(dist_to_dest[n] * model.y[H, n] for n in model.N),
        sense=pyo.minimize
    )

    # --- Constraints ---
    current_pos = current_state_info['current_pos']
    start_node_t0 = min(env_info['node_positions'].keys(),
                        key=lambda name: haversine(current_pos[0], current_pos[1], env_info['node_positions'][name][0], env_info['node_positions'][name][1]))

    # A. Initial State
    model.initial_location_con = pyo.Constraint(model.N, rule=lambda m, n: m.y[0, n] == (1 if n == start_node_t0 else 0))
    model.initial_battery_con = pyo.Constraint(rule=lambda m: m.battery[0] == current_state_info['battery_pct'])

    # B. State Transitions & Movement
    model.location_transition_con = pyo.Constraint(model.T, model.N,
        rule=lambda m, t, n: m.y[t+1, n] == sum(m.x[t, i, n] for i,j in m.E if j==n) + (m.y[t, n] - sum(m.x[t, i, j] for i,j in m.E if i==n)))
    model.must_be_somewhere_con = pyo.Constraint(model.T_plus_1, rule=lambda m, t: sum(m.y[t, n] for n in m.N) == 1)
    model.can_only_move_con = pyo.Constraint(model.T, model.N, rule=lambda m, t, n: sum(m.x[t, i, j] for i,j in m.E if i==n) <= m.y[t, n])
    model.movement_or_staying_con = pyo.Constraint(model.T, rule=lambda m, t: sum(m.x[t, i, j] for i,j in m.E) <= 1)

    # C. Consequence Constraints (Distance, Energy, Traffic)
    model.path_dist_con = pyo.Constraint(model.T, rule=lambda m, t: m.path_dist[t] == sum(env_info['distances'][i, j] * m.x[t, i, j] for i,j in m.E))
    model.path_energy_con = pyo.Constraint(model.T, rule=lambda m, t: m.path_energy[t] == sum(env_info['energy_costs'][i, j] * m.x[t, i, j] for i,j in m.E))
    M = 1e4
    model.max_traffic_con = pyo.Constraint(model.T, model.C,
        rule=lambda m, t, c: m.max_traffic[t] >= env_info['traffic_predictor'].predict(c, t) - M * (1 - sum(m.x[t, i, c] for i in m.N if i != c)))

    # D. Battery and charging logic
    charge_per_step = env_info['increase_rate'] * env_info['step_size']

    def battery_update_rule(m, t):
        # Charge gained is a fixed amount if the EV is at a charger, otherwise it's 0.
        # `is_charging` is 1 if y[t,c] is 1 for any charger c.
        is_charging = sum(m.y[t, c] for c in m.C)
        charge_gained = charge_per_step * is_charging

        # The new battery level cannot exceed 100%.
        # The `min` function here is handled implicitly by the variable's upper bound (100.0).
        return m.battery[t+1] == m.battery[t] - m.path_energy[t] + charge_gained
    model.battery_update_con = pyo.Constraint(model.T, rule=battery_update_rule)

    def battery_feasibility_rule(m, t):
        # The battery level at the end of any future step must be above the minimum.
        return m.battery[t] >= 0.0
    model.battery_feasibility_con = pyo.Constraint(pyo.RangeSet(1, H), rule=battery_feasibility_rule)

    # --- Solve ---
    results = solver.solve(model, tee=False)
    
    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        
        # Get an ordered list of charger names to map to indices
        sorted_chargers = sorted(env_info['chargers'])
        
        optimal_charger_path = []

        # Iterate through the entire time horizon of the plan
        for t in model.T:
            # Find which move was made at this timestep
            for i, j in model.E:
                if pyo.value(model.x[t, i, j]) > 0.5:
                    # A move from node i to node j was made.
                    # Check if the destination 'j' is a charger.
                    if j in sorted_chargers:
                        # Convert the charger name to a 1-based index
                        charger_idx = sorted_chargers.index(j) + 1
                        
                        # Add the charger to the path if it's not a repeat of the last stop
                        if not optimal_charger_path or optimal_charger_path[-1] != charger_idx:
                            optimal_charger_path.append(charger_idx)
                    break # Found the move for this timestep, so exit inner loop
        
        return optimal_charger_path

    else:
        # If the solver fails, return an empty list, which means "go straight to destination".
        print(f"MPC solver failed (Status: {results.solver.status}, Term: {results.solver.termination_condition}). Defaulting to direct path.")
        return []
