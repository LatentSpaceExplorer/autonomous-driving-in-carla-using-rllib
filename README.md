# CARLA Deep Reinforcement Learning with Ray RLlib

This project provides a framework for training reinforcement learning agents in the CARLA simulator using Ray RLlib. It includes a custom CARLA environment focused on lane centering (`RoadCenterEnv`) and configurations for DQN and PPO algorithms.

## Features

*   **CARLA Integration:** Simulates realistic driving scenarios.
*   **Ray RLlib:** Utilizes Ray's distributed training capabilities.
*   **Custom Gymnasium Environment:** `RoadCenterEnv` for a lane-centering task, using LiDAR and vehicle state as observations.
*   **Frame Stacking:** Uses multiple past observations to provide temporal context to the agent.

## Prerequisites

*   **CARLA Simulator:** Version 0.9.14.
    *   Download from [CARLA Releases](https://github.com/carla-simulator/carla/releases).
    *   Ensure the CARLA server is running before starting training.
*   **UV (or pip):** For installing Python dependencies.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/LatentSpaceExplorer/autonomous-driving-in-carla-using-rllib.git
    ```

2. **Download CARLA:**
    Download CARLA 0.9.14 from [CARLA Releases](https://github.com/carla-simulator/carla/releases).

3. **Install Python dependencies using [UV](https://github.com/astral-sh/uv):**
    ```bash
    uv sync
    ```

## Running the Project

1. **Start the CARLA Simulator:**
    Navigate to your CARLA installation directory and run:
    ```bash
    ./CarlaUE4.sh
    ```

2. **Run the training script:**
    Open a new terminal, navigate to the project directory, and run `train.py`:
    ```bash
    uv run train.py [OPTIONS]
    ```

    **Command-line Options:**

    *   `-c, --cpus CPUS`: Number of CPUs to use (default: 4)
    *   `-g, --gpus GPUS`: Number of GPUs to use (default: 0)
    *   `-e, --env ENV`: Environment to use (default: RoadCenterEnv)
    *   `-a, --algo ALGO`: Algorithm to use (DQN or PPO, default: DQN)
    *   `-d, --debug`: Run in debug mode
    *   `-i, --iterations ITERATIONS`: Number of training iterations (default: 100)
    *   `-m, --max_episode_steps MAX_EPISODE_STEPS`: Maximum number of steps per episode (default: 1000)

    **Example:**
    ```bash
    uv run train.py --env RoadCenterEnv --algo DQN --cpus 8 --gpus 1 --debug --iterations 500
    ```

3.  **Monitor with TensorBoard:**
    The script will automatically launch TensorBoard at `http://localhost:6006`.

## Configuration

*   **Algorithm Hyperparameters:**
    *   DQN: Modify `algos/dqn.py`
    *   PPO: Modify `algos/ppo.py`
*   **Environment Parameters:**
    *   Modify `envs/road_center_env.py` 
<!-- *   **RLlib Trainable:** -->


## `RoadCenterEnv` Details

*   **Goal:** Keep the ego vehicle centered in its lane while maintaining a target speed while avoiding collisions.
*   **Observation Space (stacked):**
    Concatenation of `num_stack_frames` of the following:
    1.  `position_diff`: Lateral offset from the lane center (meters).
    2.  `angle_diff`: Difference in yaw angle with the lane direction (degrees).
    3.  `vehicle_speed`: Current speed of the vehicle (Km/h).
    4.  `last_steer`: Previous steering command.
    5.  `last_throttle`: Previous throttle command.
    6.  `lidar_distances`: `reduced_lidar_sections` (e.g., 36) distance readings from a 360-degree planar LiDAR.
*   **Action Space:** Discrete. Mapped to continuous in `utils/actions.py`.
*   **Reward Function:**
    *   Positive reward for being close to the lane center.
    *   Positive reward for being close to a target speed (with penalty for exceeding it).
    *   Penalty for illegal stops (stopping without an obstacle).
*   **Termination Conditions:**
    *   Collision detected.
