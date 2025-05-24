import carla
import numpy as np
from gym import spaces
from typing import List


class LidarSensor():

    def __init__(self, world, buffer_size: int = 1, lidar_range: int = 50, channels: int = 32,
                 points_per_second: int = 1000, rotation_frequency: float = 10.0, upper_fov: int = 10,
                 lower_fov: int = -30, height: float = 3.0):
        self.world = world
        self.buffer_size = buffer_size
        self.range = lidar_range
        self.channels = channels
        self.points_per_second = points_per_second
        self.rotation_frequency = rotation_frequency
        self.upper_fov = upper_fov
        self.lower_fov = lower_fov
        self.height = height
        self.buffer = self.init_buffer()

    def init_buffer(self) -> List[np.ndarray]:
        return [np.array([0.0] * self.channels)
                for _ in range(self.buffer_size)]

    def create(self, ego_vehicle):
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(self.range))
        lidar_bp.set_attribute('channels', str(self.channels))
        lidar_bp.set_attribute('points_per_second', str(self.points_per_second))
        lidar_bp.set_attribute('rotation_frequency', str(self.rotation_frequency))
        lidar_bp.set_attribute('upper_fov', str(self.upper_fov))
        lidar_bp.set_attribute('lower_fov', str(self.lower_fov))
        transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=self.height))
        self.sensor = self.world.spawn_actor(lidar_bp, transform, attach_to=ego_vehicle)
        self.sensor.listen(self._callback)
        return self.sensor

    def get_observation_space(self) -> spaces.Space:
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.channels,),
            dtype=np.float32
        )

    def get_observation(self) -> np.ndarray:
        measurement = self.sensor_data
        self._add_to_buffer(measurement)
        return np.array(self.buffer, dtype=np.float32)

    def _callback(self, data):
        points = np.frombuffer(data.raw_data, dtype=np.float32)
        points = np.reshape(points, (-1, 4)) # 4 values: x, y, z, intensity
        self.sensor_data = points

    def _add_to_buffer(self, measurement):
        self.buffer.append(measurement)
        while len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)


def attach_lidar_sensor(world, vehicle, lidar_transform, callback, range=10, rotation_frequency=10, upper_fov=10,
                        lower_fov=-30):
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')

    # Set LiDAR attributes
    # lidar_bp.set_attribute('points_per_second', str(points_per_second))
    # lidar_bp.set_attribute('channels', str(channels))
    lidar_bp.set_attribute('range', str(range))
    lidar_bp.set_attribute('rotation_frequency', str(rotation_frequency))
    lidar_bp.set_attribute('upper_fov', str(upper_fov))
    lidar_bp.set_attribute('lower_fov', str(lower_fov))

    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar_sensor.listen(callback)
    return lidar_sensor


def filter_lidar_points(all_points, lidar_height):

    # filter below vehicle
    collision_height = -lidar_height + 0.5
    collision_points = all_points[all_points[:, 2] > collision_height]

    # filter above lidar
    collision_height = lidar_height
    collision_points = collision_points[collision_points[:, 2] < collision_height]

    # filter out points that are too close to the vehicle
    collision_points = collision_points[
        (abs(collision_points[:, 0]) > 2.5) |
        (abs(collision_points[:, 1]) > 1.0)
    ]

    return collision_points


def reduce_points_to_360_degrees(points):

    # reduce the data to 360 degrees (closest point to the ego vehicle)
    degree_points = {}
    for point in points:
        x, y, z, intensity = point
        # Calculate angle in degrees (0-359)
        angle = int((np.degrees(np.arctan2(y, x)) + 360) % 360)
        
        # Calculate distance to point
        distance = np.sqrt(x*x + y*y)
        
        # Store the point if it's closer than the existing point for this angle
        if angle not in degree_points or distance < degree_points[angle][1]:
            degree_points[angle] = (point, distance)

    # Convert to numpy array for plotting (keep only the points, not the distances)
    return np.array([point_data[0] for point_data in degree_points.values()])


def get_distances_360_degrees(points, default_distance=20.0):
    # Return default array if points is empty
    if len(points) == 0:
        return np.full(360, default_distance)
    
    # Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    
    # Calculate angles and distances in vectorized form
    angles = ((np.degrees(np.arctan2(y, x)) + 360) % 360).astype(int)
    distances = np.sqrt(x*x + y*y)
    
    # Initialize output array
    result = np.full(360, default_distance)
    
    # Use numpy's minimum.at to find minimum distances for each angle
    np.minimum.at(result, angles, distances)
    
    return result


def get_num_distances_360_degrees(points, default_distance=20.0, num_sectors=36):
    # Return default array if points is empty
    if len(points) == 0:
        return np.full(num_sectors, default_distance)
    
    # Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    
    # Calculate angles and distances in vectorized form
    angles = ((np.degrees(np.arctan2(y, x)) + 360) % 360).astype(int)

    # Convert 360-degree angles to sector indices (0 to num_sectors-1)
    sector_indices = angles // (360 // num_sectors)
    
    distances = np.sqrt(x*x + y*y)
    
    # Initialize output array for 36 sectors
    result = np.full(num_sectors, default_distance)
    
    # Use numpy's minimum.at to find minimum distances for each sector
    np.minimum.at(result, sector_indices, distances)
    
    return result