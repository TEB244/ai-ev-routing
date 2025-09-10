import numpy as np
import torch

try:
    from pyvrp import Model, ProblemData, Client, VehicleType, Depot
    from pyvrp.stop import MaxRuntime
except ImportError:
    raise ImportError("PyVRP is not installed. Please install it using 'pip install pyvrp'.")

def generate_path_with_pyvrp(environment, agent_index: int):
    """
    Uses PyVRP to compute an optimal route for an agent and returns the path
    as a list of node indices.
    """
    node_locations = environment.get_node_locations_for_agent(agent_index)
    distance_matrix, traffic_matrix = environment.get_costs_for_agent(agent_index, node_locations)

    num_nodes = len(node_locations)
    num_clients = num_nodes - 1 

    cost_matrix = np.round((0.5 * distance_matrix + 0.5 * traffic_matrix) * 100).astype(int)
    
    # The cost from a location to itself must be 0.
    np.fill_diagonal(cost_matrix, 0)

    duration_matrix = np.zeros_like(cost_matrix)

    COORD_SCALING_FACTOR = 10000

    vehicle_type = VehicleType(capacity=num_clients, num_available=1)
    
    depot_loc = node_locations[0]
    depot = Depot(x=int(depot_loc[1] * COORD_SCALING_FACTOR), 
                  y=int(depot_loc[0] * COORD_SCALING_FACTOR))
    
    clients = [
        Client(x=int(loc[1] * COORD_SCALING_FACTOR),
               y=int(loc[0] * COORD_SCALING_FACTOR),
               delivery=1)
        for loc in node_locations[1:]
    ]

    problem_data = ProblemData(
        clients=clients,
        depots=[depot],
        vehicle_types=[vehicle_type],
        distance_matrices=[cost_matrix],
        duration_matrices=[duration_matrix]
    )

    model = Model.from_data(problem_data)
    
    stop = MaxRuntime(2.0)
    result = model.solve(stop=stop, display=False)

    if result.best.is_feasible() and len(result.best.routes()) > 0:
        pyvrp_route = result.best.routes()[0]
        optimal_client_route_list = list(pyvrp_route)
        preset_path = [0] + optimal_client_route_list
    else:
        print(f"PyVRP solver failed for agent {agent_index}. Falling back to direct path.")
        preset_path = [0, num_nodes - 1]
        
    return preset_path