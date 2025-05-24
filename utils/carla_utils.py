from typing import Optional, Any

import carla
import numpy as np
import random   
import math



def get_client(host: str = '127.0.0.1', port: int = 2000, timeout: float = 60.0):
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    return client


def init_world(client: Any, world_name: Optional[str] = None):
    return (client.get_world()
        if world_name is None
        else client.load_world(world_name))


def set_synchronous_mode(world, time_step: float = 0.1):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = time_step # time_step=0.1 -> 10fps
    world.apply_settings(settings)


def destroy_all_vehicles(world):
    all_actors = world.get_actors()
    vehicle_actors = all_actors.filter('vehicle.*')

    for vehicle in vehicle_actors:
        vehicle.destroy()


def spawn_hero(world, spawn_point, vehicle_type='vehicle.lincoln.mkz_2020'):
    hero_bp = world.get_blueprint_library().find(vehicle_type)
    hero_bp.set_attribute('role_name', 'hero')
    return world.spawn_actor(hero_bp, spawn_point)


def spawn_npc(world, spawn_point, enable_autopilot=True):
    vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
    vehicle = world.try_spawn_actor(random.choice(vehicle_blueprints), spawn_point)

    # print(f"Spawned NPC: {vehicle}")
    if enable_autopilot and vehicle is not None:
        vehicle.set_autopilot(True)


def update_spectator(world, transform):
    # Get the spectator
    spectator = world.get_spectator()

    # # Set the spectator's position behind and above the vehicle
    distance = 4.0
    high = 2.25
    spectator_location = transform.location + carla.Location(x=-distance * np.cos(np.radians(transform.rotation.yaw)),
                                                                y=-distance * np.sin(np.radians(transform.rotation.yaw)),
                                                                z=high)
    spectator_rotation = carla.Rotation(pitch=-20, yaw=transform.rotation.yaw, roll=0.0)
    spectator_transform = carla.Transform(spectator_location, spectator_rotation)

    # Set the spectator's transform
    spectator.set_transform(spectator_transform)


def debug_text(world, text_location, text, color=carla.Color(255, 0, 0), duration=0.01):

    world.debug.draw_string(
        location=text_location,
        text=text,
        life_time=duration,
        draw_shadow=True,
        color=carla.Color(255, 0, 0)  # Red text
    )


def get_world(client, town_name=None):
    if town_name is None:
        return client.get_world()
    else:
        return client.load_world(town_name)


def enable_synchronous_mode(world, traffic_manager, fixed_delta_seconds=0.1):
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enable synchronous mode
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)

    traffic_manager.set_synchronous_mode(True)

    return world


def disable_synchronous_mode(world, traffic_manager):
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)

    traffic_manager.set_synchronous_mode(False)

    return world


def get_vehicle_speed(vehicle):
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)