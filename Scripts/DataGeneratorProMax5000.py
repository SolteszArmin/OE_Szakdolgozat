'''Ez egy template fálj, bármilyen Carla script alpaja kb. ez.'''
#Carla is very sensitive to the Python version on Windows. The recommended version is 3.7.
#################################################################################################################
#Carla Eviroment setup
import glob
import os
import math
import sys
import time
import string
import json
import random
import traceback
import carla
from collections import defaultdict
import pygame
import numpy as np
from PIL import Image

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
###################################################################################################################



host_ip='localhost'
base_port=2000
frame_number=0
actor_list = []
jsonDict=defaultdict(lambda: defaultdict(lambda:{}))
labelDict=defaultdict(lambda:{})
client= carla.Client(host_ip,base_port)
client.set_timeout(10.0)      
world=client.get_world()
carla_map = world.get_map()


def strip_world(world):
    world_actors=world.get_actors()
    
    for actor in world_actors:
       
        if actor.type_id.startswith('traffic') or 'prop' in  actor.type_id:
            actor.destroy()
def generate_random_code(len):
    ascii_letters = string.ascii_letters
    return ''.join(random.choices(ascii_letters, k=len))


code=generate_random_code(4)
data_path=f"dataGeneration/output/data/{code}/exp_{code}"
os.makedirs(f"dataGeneration/output/data/{code}",exist_ok=True)
with open(f"exp_{code}.json","w") as file:
    ed={}
    json.dump(ed,file,indent=4)

def capture_and_save_image(camera, file_name, output_folder="dataGeneration/output/pictures"):
    """
    Captures an image from a CARLA camera object and saves it to a specified folder.
    
    :param camera: The CARLA camera sensor object.
    :param file_name: The name of the output image file (without extension).
    :param output_folder: The folder where the image will be saved.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_saved = [False]

    def process_image(image):
        """
        Callback function to process and save the captured image.

        :param image: The CARLA image object.
        """
        # Convert image data to a numpy array (BGRA format)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))

        # Convert to RGB format (ignoring alpha channel)
        rgb_array = array[:, :, [2, 1, 0]] 

        # Save the image using Pygame
        output_path = os.path.join(output_folder, f"{file_name}.png")
        pygame.image.save(pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1)), output_path)

        print(f"Image saved: {output_path}")
        image_saved[0] = True  # Mark the image as saved

    # Start listening for one frame
    camera.listen(lambda image: process_image(image))

    # Wait until the image is saved
        

    # Stop the camera listener
    camera.stop()
        
def get_vehicle_speed(vehicle):
    """
    Calculates the speed of a vehicle in meters per second (m/s) from its velocity vector.

    Args:
        vehicle (carla.Vehicle): The CARLA vehicle actor.

    Returns:
        float: The speed of the vehicle in m/s.
    """
    # Get the velocity vector from the vehicle
    velocity = vehicle.get_velocity()
    
    # Compute the speed (magnitude of the velocity vector)
    speed = math.sqrt(velocity.x**2 + velocity.y**2  + velocity.z**2)
    
    return round(speed*3.6,0)

def add_frame_to_json(file_path,frame):
    with open(file_path, 'a') as file:
            file.write(json.dumps(frame) + '\n')

def get_relative_position(ego_velocity, exo_velocity, ego_position, exo_position):
    """
    Calculate the relative position, direction, distance, and relative location (left/right/front/back)
    between an ego vehicle and an exo vehicle.

    Parameters:
        ego_velocity (Vector3D): Velocity vector of the ego vehicle.
        exo_velocity (Vector3D): Velocity vector of the exo vehicle.
        ego_position (Vector3D): Position vector of the ego vehicle.
        exo_position (Vector3D): Position vector of the exo vehicle.

    Returns:
        dict: A dictionary containing:
            - relative_position (Vector3D): Relative position vector from ego to exo vehicle.
            - relative_direction (Vector3D): Normalized direction vector from ego to exo vehicle.
            - distance (float): Euclidean distance between the two vehicles.
            - relative_location (str): Relative location of the exo vehicle ("front", "back", "left", "right").
    """
     # Compute relative position vector
    relative_position = carla.Vector3D(
        exo_position.x - ego_position.x,
        exo_position.y - ego_position.y,
        exo_position.z - ego_position.z
    )
    
    # Compute Euclidean distance
    distance = math.sqrt(
        relative_position.x ** 2 +
        relative_position.y ** 2 +
        relative_position.z ** 2
    )
    
    # Normalize the relative position vector to get direction
    if distance != 0:
        relative_direction = carla.Vector3D(
            relative_position.x / distance,
            relative_position.y / distance,
            relative_position.z / distance
        )
    else:
        # If the distance is zero, direction is undefined; return zero vector
        relative_direction = carla.Vector3D(0.0, 0.0, 0.0)
    
    # Compute relative velocity vector
    relative_velocity = carla.Vector3D(
        exo_velocity.x - ego_velocity.x,
        exo_velocity.y - ego_velocity.y,
        exo_velocity.z - ego_velocity.z
    )
     
    # Determine movement direction (towards/away)
    dot_product = (
        relative_velocity.x * relative_position.x +
        relative_velocity.y * relative_position.y +
        relative_velocity.z * relative_position.z
    )
    movement_direction = "towards" if dot_product < 0 else "away"
    
    # Determine relative location (front/back/left/right)
    relative_location = ""
    if relative_position.x > 0:
        relative_location += "front"
    elif relative_position.x < 0:
        relative_location += "back"
    
    if relative_position.y > 0:
        relative_location += " right" if relative_location else "right"
    elif relative_position.y < 0:
        relative_location += " left" if relative_location else "left"
    
    # Clean up the relative location string
    relative_location = relative_location.strip()
    
    # Return results as a dictionary
   
    return relative_position, relative_direction,distance,relative_location,movement_direction

def nex_waypoint(p1,p2):
     # Compute relative position vector
    relative_position = carla.Vector3D(
        p2.x - p1.x,
        p2.y - p1.y,
        p2.z - p1.z
    )
    
    # Compute Euclidean distance
    distance = math.sqrt(
        relative_position.x ** 2 +
        relative_position.y ** 2 +
        relative_position.z ** 2
    )
    
    NONE_TRESHOLD=3
    SLIGHT_THRESHOLD = 15
    HARD_THRESHOLD = 75
    angle = math.degrees(math.atan2(relative_position.y, relative_position.x))
    angle = abs(angle)  # Only care about the magnitude for this classification
    

    if angle<=NONE_TRESHOLD:
        intensity="straight"
    elif angle <= SLIGHT_THRESHOLD:
        intensity = "slight"
    elif angle >= HARD_THRESHOLD:
        intensity = "hard"
    else:
        intensity = "medium"
    relative_location = []
    if intensity!='straight':

        if relative_position.y > 0:
            relative_location.append("right")
        elif relative_position.y < 0:
            relative_location.append("left")
        else:
            intensity="straight"
    if relative_position.x > 0:
        relative_location.append("front")
    elif relative_position.x < 0:
        relative_location.append("back")
    primary_location = " ".join(relative_location)
        
    if intensity:
        primary_location=f"{intensity} {primary_location}"

    
    return primary_location

def ego_data():
    global ego
    ego_pos = ego.get_location()
    ego_vel=ego.get_velocity()
    ego_spd=get_vehicle_speed(ego)
    map=world.get_map()
    current_waypoint=map.get_waypoint(ego_pos)
    next_waypoint = current_waypoint.next(4.0)[0]
    transform=ego.get_transform()
    Lanechange=False
    lane_type = current_waypoint.lane_type if current_waypoint else "Unknown"
    Turn=False
    if current_waypoint.lane_id != next_waypoint.lane_id:
        Lanechange=True 
    if current_waypoint.road_id != next_waypoint.road_id:
        Turn=True

    if ego.is_at_traffic_light():
        traffic_light_state = ego.get_traffic_light().get_state()
    else:
        traffic_light_state=None
    junction=next_waypoint.is_junction
    dir=nex_waypoint(ego_pos,next_waypoint.transform.location)
    position = {
                "x":ego_pos.x, 
                "y":ego_pos.y,
                "z":ego_pos .z
            }
    waypoint={
        "x":next_waypoint.transform.location.x,
        "y":next_waypoint.transform.location.y,
        "z":next_waypoint.transform.location.z

    }
    velocity_vector = {
                "x":ego_vel.x,
                "y":ego_vel.y,
                "z":ego_vel.z
            }
    current_rotation= {
                "pitch": transform.rotation.pitch,
                "yaw": transform.rotation.yaw,
                "roll": transform.rotation.roll
            }
    return{        
        "ID": "EGO",
        "Type": "EGO",
        "Traffic_Light_State": str(traffic_light_state),
        "Distance_from_ego":0,
        "Rotation":current_rotation,
        "Lane": str(lane_type),
        "Position": position,
        "Velocity": velocity_vector,
        "Dimensions": None,
        "relative_position":None, 
        "relative_direction":None, 
        "relative_location":None,
        "relative_movement_direction":None,
        "position":position,
        "waypoint_location":waypoint,
        "speed":ego_spd,
        "Lanechange":Lanechange,
        "Turn":Turn,
        "Next_direction":dir,
        "Is_Junction":junction
    }

def process_exo_vehicles(snapshot):
    global frame_number
    global ego
    global camera

    ego_loc = ego.get_location()
    ego_vel = ego.get_velocity()
    vehicles = world.get_actors().filter('vehicle.*')
    
    # Ensure jsonDict[frame_number] exists
    

    for vehicle in vehicles:

        relative_pos, relative_dir,distance,relative_location,movement_direction =get_relative_position(ego_loc,ego_vel, vehicle.get_location(),vehicle.get_velocity())
        
        if distance < 40:    

            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            speed = get_vehicle_speed(vehicle)
            bounding_box = vehicle.bounding_box
            vehicle_type = vehicle.type_id

            traffic_light_state = None
            if vehicle.is_at_traffic_light():
                traffic_light_state = vehicle.get_traffic_light().get_state()

            waypoint = world.get_map().get_waypoint(vehicle.get_location())
            lane_type = waypoint.lane_type if waypoint else "Unknown"

            current_rotation= {
                "pitch": transform.rotation.pitch,
                "yaw": transform.rotation.yaw,
                "roll": transform.rotation.roll
            }

            relative_position={
                "x": relative_pos.x,
                "y": relative_pos.y,
                "z": relative_pos.z}, 
            
            relative_direction = {
                "x": relative_dir.x,
                "y": relative_dir.y,
                "z": relative_dir.z
            }

            bounding_box_dims = {
                "extent_x":bounding_box.extent.x, 
                "extent_y":bounding_box.extent.y,
                "extent_z":bounding_box.extent.z
            }
            position = {
                "x":transform.location.x, 
                "y":transform.location.y,
                "z":transform.location.z
            }
            velocity_vector = {
                "x":velocity.x,
                "y":velocity.y,
                "z":velocity.z
            }

            if frame_number%20==0:
                #capture_and_save_image(camera, frame_number)
                jsonDict[f"frame {frame_number}"][f"vehicle EGO"]=ego_data()
                jsonDict[f"frame {frame_number}"][f"vehicle {vehicle.id}"] = {
                    "ID": vehicle.id,
                    "Type": vehicle_type,
                    "Traffic_Light_State": str(traffic_light_state),
                    "Distance_from_ego":distance,
                    "Speed": speed,
                    "Rotation":current_rotation,
                    "Lane": str(lane_type),
                    "Position": position,
                    "Velocity": velocity_vector,
                    "Dimensions": bounding_box_dims, 
                    "relative_position":relative_position, 
                    "relative_direction":relative_direction, 
                    "relative_location":relative_location,
                    "relative_movement_direction":movement_direction,
                    "position":position,
                    "waypoint_location":None,
                    "Lanechange":None,
                    "Turn":None,
                    "Next_direction":None,
                    "Is_Junction":None
                    }
          

    frame_number += 1

def generatecars(blueprint_library,spawn_points,number_of_vehicles,port):
    blueprints=blueprint_library.filter("vehicle.*")
    for i in range(number_of_vehicles):
        blueprint = random.choice(blueprints)
        sp=random.choice(spawn_points)
        try:
            car=world.spawn_actor(blueprint, sp)
            car.set_autopilot(True,port)
            actor_list.append(car)
        except Exception as e:
            pass

def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        # Convert defaultdict to a regular dict
        d = {k: defaultdict_to_dict(v) if isinstance(v, (defaultdict, dict)) else v for k, v in d.items()}
    return dict(d)


try:
    
    #world=client.load_world('Town05') 
    world=client.get_world()
    carla_map = world.get_map()
    blueprint_library=world.get_blueprint_library()
    #strip_world(world)
    # settings = world.get_settings()
    # settings.fixed_delta_seconds = 1.0  # Set to 1 second per tick
    # settings.synchronous_mode = False    # Enable synchronous mode
    # world.apply_settings(settings)
    #cts=CarlaTrafficSimulator()
    #cts.spawn_vehicles()
    


    bp=blueprint_library.filter("mercedes")[0]
    print(bp)
    tm = client.get_trafficmanager(base_port)
    tm_port=tm.get_port()
    import random
    import keyboard
    spawnpoints=world.get_map().get_spawn_points()
    spawn_point=random.choice(spawnpoints)
    ego=world.spawn_actor(bp, spawn_point)
    ego.set_autopilot(True,tm_port)
    actor_list.append(ego)
    cam_bp=blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x","620")
    cam_bp.set_attribute("image_size_y","480")
    cam_bp.set_attribute("fov","90")

    cam_sp = carla.Transform(carla.Location(x=2.5,z=0.7))
    generatecars(blueprint_library,spawnpoints,50,tm_port)
    camera=world.spawn_actor(cam_bp,cam_sp,attach_to=ego)
    actor_list.append(camera)
    #camera.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame))

    id=world.on_tick(process_exo_vehicles)
    while not keyboard.is_pressed('space'):
        vehicles = world.get_actors().filter('vehicle.*')
        debug = world.debug

        pass
    camera.destroy()

except Exception as e:
    print(e)
    traceback.print_exc()


finally:

    dictc=defaultdict_to_dict(jsonDict)
    
    with open(f"dataGeneration/output/data/data_{code}.json","w") as file:
        json.dump(dictc,file,indent=4) 
    world.remove_on_tick(id)
    for actor in actor_list:
         actor.destroy()
        
    print('Process finished, actors removed.')