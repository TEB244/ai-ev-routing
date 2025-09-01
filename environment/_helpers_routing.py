# Created by Lucas
# Restructure by Santiago 03/07/2024

# Helper functions to route and update to simulate routing in the environment class
import torch
import numpy as np

def get_actions(actions, stops, dtype):
    """
    Generates a new identity matrix and creates action vectors based on the stops.

    Parameters:
        actions (torch.Tensor): A tensor representing possible actions.
        stops (torch.Tensor): A tensor containing the stops for each action (assumed to be on the correct device).
        dtype (torch.dtype): The desired data type for output tensor.

    Returns:
        torch.Tensor: A tensor containing the action vectors.
    """

    # Make new identity matrix on the same device as stops
    max_val = actions.shape[1]
    identity_matrix = torch.eye(max_val, dtype=dtype, device=stops.device)

    # Create boolean matrix for indexing on the same device as stops
    left_matrix = (stops[:, 0].reshape(-1, 1) == torch.arange(max_val, dtype=stops.dtype, device=stops.device)) # Use stops.dtype for arange comparison
    # Perform matmul and convert result to int
    actions = torch.matmul(left_matrix.to(dtype), identity_matrix).int()

    # removing unused tensors from memory
    identity_matrix = None
    left_matrix = None

    return actions.to(dtype)

def get_distance(tokens, destinations, actions):
    """
    Computes the Euclidean distance between each token's position and its corresponding target destination.

    Parameters:
        tokens (torch.Tensor): A tensor containing the coordinates of the tokens.
        destinations (torch.Tensor): A tensor containing the coordinates of the possible destinations.
        actions (torch.Tensor): A tensor indicating the chosen actions for each token.

    Returns:
        torch.Tensor: A tensor containing the distances between each token and its target destination as a 1D array.
    """

    # Compute targets for each token
    token_targets = torch.matmul(actions, destinations)

    # Compute Euclidean distance for each token
    dist = ((tokens - token_targets) * torch.max(actions, axis=1).values.unsqueeze(1))
    total_dist = torch.sqrt(torch.sum(dist ** 2, axis=1)).reshape(-1, 1)

    # Return distances as Nx1 matrix
    return total_dist.reshape(-1)

def get_arrived(dists, dist_threshold):
    """
    Determines whether each distance is below or above a given threshold.

    Parameters:
        dists (torch.Tensor): A tensor containing distances.
        dist_threshold (float): The distance threshold.

    Returns:
        torch.Tensor: A tensor containing 1s and 0s where 1 indicates the distance is below or equal to the threshold
                      and 0 indicates the distance is above the threshold.
    """

    # Return 1 if below threshold, and 0 if above
    return (dists <= dist_threshold).int()

def get_traffic(stops, destinations, is_charging, traffic_noise, seed):
    """
    Calculates the traffic level at each charging station based on the stops and charging status of vehicles.

    Parameters:
        stops (torch.Tensor): A tensor containing the stops for each vehicle.
        destinations (torch.Tensor): A tensor containing the coordinates of possible destinations.
        is_charging (torch.Tensor): A tensor indicating whether each vehicle is charging (1) or not (0).
        traffic_noise (float): The noise level for the traffic.
        seed (int): The seed for the random number generator.
    Returns:
        torch.Tensor: A tensor containing the traffic level at each charging station.
    """

    torch.manual_seed(seed)

    # Get stations to charge at
    target_stations = stops[:, 0].reshape(-1)

    # Isolate stations to charge at
    # Ensure tensor(0) is created on the same device
    stations_in_use = (target_stations * is_charging).int()
    zero_tensor = torch.tensor(0, dtype=stations_in_use.dtype, device=stations_in_use.device)
    stations_in_use = torch.maximum(stations_in_use -1, zero_tensor)

    # Used to find end of the list
    max_num = destinations.shape[0]

    # Get the traffic level at each charging station
    traffic_level = torch.bincount(stations_in_use, minlength=max_num)

    # Add noise to the traffic level
    traffic_level = traffic_level + torch.round(torch.randn(traffic_level.shape, device=traffic_level.device) * traffic_noise)

    # Ignore destination 0 if it's not a valid station ID
    if max_num > 0:
        traffic_level[0] = 0

    # Note that the first N entries can be ignored since they aren't charging stations, but rather final destinations
    return traffic_level

def get_charging_rates(stops, traffic_level, arrived, capacity, decrease_rates, increase_rate, dtype):
    """
    Calculates the charging rates for each vehicle based on their current stops, traffic level at charging stations,
    arrival status, and station capacities.

    Parameters:
        stops (torch.Tensor): A tensor containing the stops for each vehicle.
        traffic_level (torch.Tensor): A tensor containing the traffic level at each charging station.
        arrived (torch.Tensor): A tensor indicating whether each vehicle has arrived (1 if arrived, 0 otherwise).
        capacity (torch.Tensor): A tensor containing the capacity of each charging station.
        decrease_rates (torch.Tensor): A tensor containing the rate at which charging decreases by model.
        increase_rate (float): The maximum rate at which charging can increase.
        dtype (torch.dtype): The desired data type for intermediate calculations.

    Returns:
        torch.Tensor: A tensor containing the charging rates for each vehicle.
    """

    # Get target stop for each car
    target_stop = stops[:, 0]

    # Get charging rate for each station
    # Ensure tensor(1.0) is created on the same device
    one_tensor = torch.tensor(1.0, dtype=traffic_level.dtype, device=traffic_level.device)
    max_traffic_level = torch.maximum(traffic_level, one_tensor)
    capacity_rate = (capacity / max_traffic_level) * increase_rate
    # Ensure increase_rate tensor is on the same device
    increase_rate_tensor = torch.tensor(increase_rate, dtype=capacity_rate.dtype, device=capacity_rate.device)
    station_rate = torch.minimum(capacity_rate, increase_rate_tensor)

    # Offset everything by decrease rate
    station_rate = station_rate.unsqueeze(1) + decrease_rates.unsqueeze(0)

    # Get charging rates for each car based on target stop index
    car_indices = torch.arange(len(decrease_rates), device=target_stop.device)
    rates_by_car = station_rate[target_stop.long(), car_indices.long()] # Use .long() for indexing

    # Zero out charging rate for cars that haven't arrived
    rates_by_car *= arrived.int()

    # Undo the offset
    rates_by_car -= decrease_rates

    # Zero-out charging rate for cars already at their final destination (assuming stop value -1 indicates final destination)
    diag_values = torch.tensor([0 if x == -1 else 1 for x in target_stop], dtype=dtype, device=capacity.device)
    diag_matrix = torch.diag(diag_values)
    rates_by_car = torch.matmul(rates_by_car.to(dtype), diag_matrix.to(dtype))

    # Cleaning unused tensors
    capacity_rate = None
    station_rate = None
    diag_matrix = None
    diag_values = None
    one_tensor = None
    increase_rate_tensor = None
    car_indices = None

    # Return Nx1 charging rate for each car
    return rates_by_car

def update_battery(battery, charge_rate, arrived_at_final):
    """
    Updates the battery levels of vehicles based on the charging rates.

    Parameters:
        battery (torch.Tensor): A tensor containing the current battery levels.
        charge_rate (torch.Tensor): A tensor containing the charging rates.
        arrived_at_final (torch.Tensor): A tensor of 1s and 0s where 1 indicates the car has reached its final destination.

    Returns:
        torch.Tensor: A tensor containing the updated battery levels.
    """
    # Create mask: 0 if arrived, 1 otherwise
    mask = (arrived_at_final.logical_not()).to(charge_rate.dtype) # Use logical_not and ensure dtype matches

    # Apply mask to charge_rate and update battery
    return battery + (charge_rate * mask.squeeze(0))


def get_battery_charged(battery, target, device, ev_rationality, seed):
    """
    Determines whether each vehicle's battery level is sufficient to depart from the charging station.

    Parameters:
        battery (torch.Tensor): A tensor containing the current battery levels.
        target (torch.Tensor): A tensor containing the target battery levels required.
        device (torch.device): Target device (used for error printing).
        ev_rationality (float): The rationality of the EV.
        seed (int): The seed for the random number generator.
    Returns:
        torch.Tensor: A tensor containing 1s and 0s where 1 indicates battery is sufficient, 0 otherwise.
    """

    # Return if the battery level is above the needed level to depart from charging station
    try:

        torch.manual_seed(seed)

        # Get random numbers between 0 and 1
        random_nums = torch.rand(battery.shape, device=device)
        # If random number is greater than ev_rationality, the EV will keep charging
        irrational_check = random_nums < ev_rationality    
        # Compare battery level with the target for the *current* stop (index 0 after updates)
        return (battery >= target[:, 0]).int() * irrational_check.int()
    except Exception as e:
        print(f"Error in get_battery_charged on device {device}: {e}")
        # It might be helpful to print shapes and dtypes here for debugging
        print(f"Battery shape: {battery.shape}, dtype: {battery.dtype}, device: {battery.device}")
        print(f"Target shape: {target.shape}, dtype: {target.dtype}, device: {target.device}")
        raise Exception(e)


def move_tokens(tokens, moving, actions, destinations, step_size):
    """
    Moves tokens towards their target destinations, updating their positions and calculating the distance traveled.

    Parameters:
        tokens (torch.Tensor): Current positions.
        moving (torch.Tensor): Whether each token is moving (1 if moving, 0 otherwise).
        actions (torch.Tensor): Chosen actions for each token.
        destinations (torch.Tensor): Possible destinations coordinates.
        step_size (float): Maximum step size for each movement.

    Returns:
        tuple: A tuple containing:
            - new_tokens (torch.Tensor): Updated positions.
            - distance_travelled (torch.Tensor): Distances traveled.
    """

    # Get target destinations for tokens that are moving
    target_coords = torch.matmul(actions, destinations)

    # Compute the displacement vectors and their norms
    displacement = torch.mul((target_coords - tokens), torch.max(actions, axis=1).values.unsqueeze(1))
    distances_to_travel = torch.linalg.norm(displacement, axis=1)

    # Compute the normalized direction vectors
    # Initialize the output tensor with zeros, matching the shape and device of displacement
    direction = torch.zeros_like(displacement)
    # Create a mask where distances are not zero
    mask = distances_to_travel != 0
    # Perform the division where the mask is True, handle potential division by zero implicitly by mask
    direction[mask] = displacement[mask] / distances_to_travel[mask].unsqueeze(1)

    # Zero-out direction for tokens that aren't moving
    masked_direction = torch.mul(direction, moving.unsqueeze(-1))

    # Compute the step vectors
    # Ensure step_size tensor is created on the same device
    step_size_tensor = torch.tensor(step_size, dtype=distances_to_travel.dtype, device=distances_to_travel.device)
    steps_dist = torch.minimum(distances_to_travel, step_size_tensor).unsqueeze(1)
    steps_vector = steps_dist * masked_direction

    # If the car is charging (not moving) but isn't directly on the charging station, shift them onto it
    shift_to_charger_while_charging = torch.mul(displacement, (moving.logical_not()).unsqueeze(1).to(displacement.dtype)) # Use logical_not

    # Update token positions
    new_tokens = tokens + steps_vector + shift_to_charger_while_charging

    # Calculate distance travelled (use updated new_tokens for accuracy if shift happened)
    distance_travelled = torch.linalg.norm(steps_vector, axis=1) # Norm of the actual step taken

    return new_tokens, distance_travelled


def update_stops(stops, ready_to_leave, dtype, device):
    """
    Updates the stops for each vehicle based on their readiness to leave the current stop.
    Shifts the stops array to the left for vehicles that are ready.

    Parameters:
        stops (torch.Tensor): Current stops for each vehicle.
        ready_to_leave (torch.Tensor): Whether each vehicle is ready (1 if ready, 0 otherwise).
        dtype (torch.dtype): Desired data type.
        device (torch.device): Target device.

    Returns:
        torch.Tensor: Updated stops for each vehicle.
    """

    # Create transformation matrix to shift elements left
    K = stops.shape[1]
    # Ensure transform_matrix is created on the correct device
    transform_matrix = torch.zeros((K, K), dtype=dtype, device=device)
    # Populate the subdiagonal below the main diagonal with ones
    if K > 1:
        transform_matrix[torch.arange(K - 1), torch.arange(1, K)] = 1 # Shift left correctly

    # Reshape ready_to_leave for broadcasting
    ready_to_leave_mask = ready_to_leave.reshape(-1, 1).to(dtype) # Ensure correct shape and dtype

    # Apply transformation only to rows where ready_to_leave is 1
    # Calculate shifted stops (for those ready to leave)
    shifted_stops = torch.matmul(stops, transform_matrix)
    # Keep original stops for those not ready
    not_ready_mask = 1.0 - ready_to_leave_mask
    updated_stops = (shifted_stops * ready_to_leave_mask) + (stops * not_ready_mask)

    # Check for infinite values which might indicate an issue elsewhere
    if torch.any(torch.isinf(updated_stops[:, 0])): # Check first column after potential shift
        print(f"Original Stops:\n{stops}")
        print(f"Ready to leave:\n{ready_to_leave}")
        print(f"Updated Stops before check:\n{updated_stops}")
        raise Exception("Infinite value encountered in stops after update!")

    return updated_stops