import random
import math
import numpy as np
import carla
import gymnasium as gym
import gymnasium.spaces as spaces
from typing import Optional

import utils.carla_utils as carla_utils
import sensors.lidar as lidar_utils
import sensors.collision as collision_utils
import sensors.obstacle as obstacle_utils
import utils.actions as actions_utils


class RoadCenterEnv(gym.Env):

    def __init__(self, env_config: Optional[dict] = None, debug: bool = False):
        super().__init__()

        # ==== Configuration ====
        self.fixed_delta_seconds = 0.1
        self.max_stopped_steps = 300
        self.num_npcs = 30
        self.lidar_range = 40.0
        self.lidar_height = 0.5
        self.reduced_lidar_sections = 36
        self.num_stack_frames = 3
        self.obstacle_spawned = False
        self.DEBUG = debug

        # ==== CARLA Setup ====
        self.client = carla_utils.get_client('127.0.0.1', 2000)
        self.world = carla_utils.get_world(self.client)
        self.traffic_manager = self.client.get_trafficmanager()
        carla_utils.enable_synchronous_mode(
            self.world, self.traffic_manager, fixed_delta_seconds=self.fixed_delta_seconds
        )

        # ==== Action and Observation Spaces ====
        self.action_space = spaces.Discrete(len(actions_utils.all_continuous_actions()))

        single_obs_low = np.array([-40.0, -720.0, -400.0, -4.0, -4.0] + [0.0] * self.reduced_lidar_sections)
        single_obs_high = np.array([40.0, 720.0, 400.0, 4.0, 4.0] + [400.0] * self.reduced_lidar_sections)
        stacked_low = np.tile(single_obs_low, self.num_stack_frames)
        stacked_high = np.tile(single_obs_high, self.num_stack_frames)
        self.observation_space = spaces.Box(
            low=stacked_low,
            high=stacked_high,
            shape=(len(single_obs_low) * self.num_stack_frames,),
            dtype=np.float32
        )

        # ==== State ====
        self.frames_buffer = []
        self.early_done_penalty = -0.0
        self.vehicle = None
        self.collision_sensor = None
        self.raw_collision_data = None
        self.obstacle_sensor = None
        self.raw_obstacle_data = None
        self.lidar_sensor = None
        self.raw_lidar_data_update_buffer = []
        self.episode_timer = 0
        self.stopped_timer = 0
        self.last_throttle = 0.0
        self.last_steer = 0.0

    def reset(self, seed=None, options=None):
        carla_utils.destroy_all_vehicles(self.world)
        self.obstacle_spawned = False

        # Destroy sensors if they exist
        for sensor in [self.collision_sensor, self.obstacle_sensor, self.lidar_sensor]:
            if sensor is not None:
                sensor.destroy()

        self.raw_collision_data = None
        self.raw_obstacle_data = None
        self.lidar_update_buffer_size = 10
        self.raw_lidar_data_update_buffer = []

        # Spawn ego vehicle and sensors
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = carla_utils.spawn_hero(self.world, random.choice(spawn_points))
        self.collision_sensor = collision_utils.attach_collision_sensor(
            self.world, self.vehicle, self._collision_callback)
        self.obstacle_sensor = obstacle_utils.attach_obstacle_sensor(
            self.world, self.vehicle, self._obstacle_callback)
        self.lidar_sensor = lidar_utils.attach_lidar_sensor(
            self.world, self.vehicle,
            carla.Transform(carla.Location(x=0.0, y=0.0, z=self.lidar_height)),
            self._lidar_callback, range=self.lidar_range
        )

        # Spawn NPC vehicles
        for _ in range(self.num_npcs):
            carla_utils.spawn_npc(self.world, random.choice(spawn_points))

        self.episode_timer = 0
        self.stopped_timer = 0
        self.last_throttle = 0.0
        self.last_steer = 0.0

        # Create initial observation
        position_diff, angle_diff = self._get_position_and_angle_difference()
        vehicle_speed = carla_utils.get_vehicle_speed(self.vehicle)

        # Wait for first Lidar data
        while len(self.raw_lidar_data_update_buffer) == 0:
            self.world.tick()

        all_points = np.vstack(self.raw_lidar_data_update_buffer)
        filtered_points = lidar_utils.filter_lidar_points(all_points, self.lidar_height)
        lidar_distances = lidar_utils.get_num_distances_360_degrees(
            filtered_points, default_distance=self.lidar_range, num_sectors=self.reduced_lidar_sections
        )

        initial_obs = np.array(
            [position_diff, angle_diff, vehicle_speed, 0.0, 0.0] + lidar_distances.tolist(),
            dtype=np.float32
        )
        self.frames_buffer = [initial_obs.copy() for _ in range(self.num_stack_frames)]
        stacked_obs = np.concatenate(self.frames_buffer)
        info = {}
        return stacked_obs, info

    def step(self, action):
        self.episode_timer += 1

        # Apply action to the vehicle
        vehicle_controls = actions_utils.get_continuous_vehicle_controls(
            int(action), self.last_throttle, self.last_steer)
        self.last_throttle = vehicle_controls.throttle
        self.last_steer = vehicle_controls.steer
        self.vehicle.apply_control(vehicle_controls)
        self.world.tick()

        # Get observations
        vehicle_speed = carla_utils.get_vehicle_speed(self.vehicle)
        position_diff, angle_diff = self._get_position_and_angle_difference()
        if len(self.raw_lidar_data_update_buffer) > 0:
            all_points = np.vstack(self.raw_lidar_data_update_buffer)
            filtered_points = lidar_utils.filter_lidar_points(all_points, self.lidar_height)
            distances_36 = lidar_utils.get_num_distances_360_degrees(
                filtered_points, default_distance=self.lidar_range, num_sectors=self.reduced_lidar_sections
            )
        else:
            distances_36 = np.array([self.lidar_range] * self.reduced_lidar_sections)

        current_obs = np.array(
            [position_diff, angle_diff, vehicle_speed, self.last_steer, self.last_throttle] + distances_36.tolist(),
            dtype=np.float32
        )
        self.frames_buffer.pop(0)
        self.frames_buffer.append(current_obs)
        obs = np.concatenate(self.frames_buffer)

        # Compute reward
        position_reward = max(0.0, (1.25 - abs(position_diff)))
        target_speed = 25.0
        speed_bonus = (min(vehicle_speed, target_speed) / target_speed) * 0.25
        speed_penalty = (max(vehicle_speed - target_speed, 0.0) / target_speed) * -0.25
        illegal_stop_penalty = -2.5 if vehicle_speed < 1.0 and self.raw_obstacle_data is None else 0.0
        if illegal_stop_penalty:
            self.raw_obstacle_data = None
        throttle_diff = abs(vehicle_controls.throttle - self.last_throttle)
        steer_diff = abs(vehicle_controls.steer - self.last_steer)
        jerky_penalty = -0.1 * (throttle_diff + steer_diff)
        reward = position_reward + speed_bonus + speed_penalty + illegal_stop_penalty + jerky_penalty

        # Done conditions
        done = self._get_done_conditions()
        truncated = False
        info = {}

        # Debug visualization
        carla_utils.update_spectator(self.world, self.vehicle.get_transform())
        if self.DEBUG:
            self._visualize_reduced_lidar(distances_36)
            text_location = self.vehicle.get_transform().location + carla.Location(x=0.0, y=0.0, z=2.0)
            carla_utils.debug_text(
                self.world, text_location,
                f"DISTANCE: {position_diff:.4f}\nANGLE_DIFF: {angle_diff:.2f} degrees\n"
                f"SHORTEST_DISTANCE: {min(distances_36):.2f}\nSPEED: {vehicle_speed:.2f} Km/h\nReward: {reward:.2f}"
            )

        return obs, reward, done, truncated, info

    # ===== Sensor Callbacks =====
    def _collision_callback(self, data):
        self.raw_collision_data = data

    def _obstacle_callback(self, data):
        self.raw_obstacle_data = data

    def _lidar_callback(self, data):
        points = np.frombuffer(data.raw_data, dtype=np.float32)
        points = np.reshape(points, (-1, 4))
        self.raw_lidar_data_update_buffer.append(points.copy())
        if len(self.raw_lidar_data_update_buffer) > self.lidar_update_buffer_size:
            self.raw_lidar_data_update_buffer.pop(0)

    # ===== Helper Functions =====
    def _get_position_and_angle_difference(self):
        vehicle_location = self.vehicle.get_transform().location
        current_waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=False)
        if current_waypoint is None:
            return 5.0, 0.0
        lane_center = current_waypoint.transform.location
        delta_vector = vehicle_location - lane_center
        lane_yaw = current_waypoint.transform.rotation.yaw
        lane_right_vector = carla.Vector3D(
            -math.sin(math.radians(lane_yaw)),
            math.cos(math.radians(lane_yaw)),
            0
        )
        lateral_offset = delta_vector.x * lane_right_vector.x + delta_vector.y * lane_right_vector.y
        lateral_offset = np.clip(lateral_offset, -5.0, 5.0)
        vehicle_yaw = self.vehicle.get_transform().rotation.yaw
        angle_diff = vehicle_yaw - lane_yaw
        angle_diff = (angle_diff + 180) % 360 - 180
        return lateral_offset, angle_diff

    def _get_done_conditions(self):
        if carla_utils.get_vehicle_speed(self.vehicle) < 0.5:
            self.stopped_timer += 1
        if self.raw_collision_data is not None:
            if self.DEBUG:
                print("Collision detected", self.raw_collision_data)
            return True
        return False

    def _visualize_reduced_lidar(self, distances_36):
        debug = self.world.debug
        transform = self.vehicle.get_transform()
        angles = np.linspace(0, 2 * np.pi, len(distances_36), endpoint=False)
        for angle, distance in zip(angles, distances_36):
            try:
                x = distance * np.cos(angle)
                y = distance * np.sin(angle)
                z = 1
                local_point = np.array([x, y, z, 1.0])
                rotation_matrix = np.array(transform.get_matrix())
                world_point = np.dot(rotation_matrix, local_point)
                location = carla.Location(world_point[0], world_point[1], world_point[2])
                debug.draw_point(location, size=0.15, color=carla.Color(0, 2, 0), life_time=0.5)
            except Exception as e:
                print(f"Error visualizing reduced point: {e}")