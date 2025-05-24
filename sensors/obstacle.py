import carla

def attach_obstacle_sensor(world, ego_vehicle, callback):
    obstacle_sensor_bp = world.get_blueprint_library().find('sensor.other.obstacle')
    obstacle_sensor_bp.set_attribute("only_dynamics", "true") # Only detect dynamic objects
    obstacle_sensor_bp.set_attribute("debug_linetrace", "true") # Draw line trace
    obstacle_sensor_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
    obstacle_sensor = world.spawn_actor(obstacle_sensor_bp, obstacle_sensor_transform, attach_to=ego_vehicle)
    obstacle_sensor.listen(callback)
    return obstacle_sensor