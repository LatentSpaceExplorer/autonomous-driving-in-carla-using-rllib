import carla

def attach_collision_sensor(world, ego_vehicle, callback):
    collision_bp = world.get_blueprint_library().find('sensor.other.collision')
    collision_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
    collision_sensor = world.spawn_actor(collision_bp, collision_transform, attach_to=ego_vehicle)
    collision_sensor.listen(callback)
    return collision_sensor