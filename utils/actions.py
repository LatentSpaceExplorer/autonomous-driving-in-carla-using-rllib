import carla    
import numpy as np


def all_continuous_actions(throttle_delta=0.1, steer_delta=0.1, brake_delta_slow=0.1, brake_delta_full=1.0):

    return {
        0: [0.0, 0.0, 0.0],   # Noop (useful for initialization or no input)
        
        1: [throttle_delta, 0.0, 0.0],   # Increase Throttle
        2: [-throttle_delta, 0.0, 0.0],  # Decrease Throttle

        3: [0.0, steer_delta, 0.0],   # Increase Steer Right
        4: [0.0, -steer_delta, 0.0],  # Increase Steer Left

        5: [0.0, 0.0, brake_delta_slow],   # Apply slow Brake 
        6: [0.0, 0.0, brake_delta_full],   # Apply full Brake 

        7: [None, None, None]     # Reset all inputs
    }

def get_continuous_vehicle_controls(action_id, current_throttle, current_steer, delta_throttle=0.1, delta_steer=0.1, brake_delta_slow=0.1, brake_delta_full=1.0):
    action_control = all_continuous_actions(delta_throttle, delta_steer, brake_delta_slow, brake_delta_full)[action_id]
    controls = carla.VehicleControl()

    if action_id == 7:  # Reset: set controls to default values
        controls.throttle = 0.0
        controls.steer = 0.0
        controls.brake = 0.0    
        controls.reverse = False
        controls.hand_brake = False
        return controls

    # Update throttle and steer based on the provided deltas, then clamp to valid ranges.
    controls.throttle = np.clip(current_throttle + action_control[0], 0.0, 1.0)  # Clamp throttle between 0 and 1.
    controls.steer = np.clip(current_steer + action_control[1], -1.0, 1.0)       # Clamp steer between -1 and 1.
    controls.brake = action_control[2]
    controls.reverse = False
    controls.hand_brake = False

    return controls



###### Old actions
def all_actions():
    return {
        0: [0.0, 0.00, 0.0, False, False],  # Coast
        1: [0.0, 0.00, 1.0, False, False],  # Apply Break
        2: [0.0, 0.75, 0.0, False, False],  # Right
        3: [0.0, -0.75, 0.0, False, False],  # Left
        4: [0.5, 0.00, 0.0, False, False],  # Medium Straight
        5: [0.5, 0.75, 0.0, False, False],  # Medium Right
        6: [0.5, -0.75, 0.0, False, False],  # Medium Left
        7: [1.0, 0.00, 0.0, False, False],  # Fast Straight
        8: [1.0, 0.75, 0.0, False, False],  # Fast Right
        9: [1.0, -0.75, 0.0, False, False],  # Fast Left
        # Reverse
        10: [-0.5, 0.00, 0.0, True, False],  # Reverse Straight
        11: [-0.5, 0.75, 0.0, True, False],  # Reverse Right
        12: [-0.5, -0.75, 0.0, True, False],  # Reverse Left
    }


def get_vehicle_controls(action_id):
    action_control = all_actions()[action_id]
    
    controls = carla.VehicleControl()
    controls.throttle = action_control[0]
    controls.steer = action_control[1]
    controls.brake = action_control[2]
    controls.reverse = action_control[3]
    controls.hand_brake = action_control[4]

    return controls
