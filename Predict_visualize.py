#Carla is very sensitive to the Python version on Windows. The recommended version is 3.7.
#################################################################################################################
#Carla Eviroment setup
import glob
import os
import sys
import httpx
import asyncio
import random
import math
import keyboard
import carla
import pygame

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
###################################################################################################################
class Prediction_maker:

    def __init__(self,host_ip,api_url,base_port):
        self.url=api_url
        self.port=base_port
        self.ip=host_ip
        self.current_frame={}
        self.current_frame_count=0
        self.actor_list = []
        
    def init_carla(self):
        self.client= carla.Client(self.ip,self.port)
        self.client.set_timeout(10.0)
        self.world=self.client.get_world()
        tm = self.client.get_trafficmanager(base_port)
        self.tm_port=tm.get_port()
        
        self.spawnpoints=self.world.get_map().get_spawn_points()

        
        self.blueprint_library=self.world.get_blueprint_library()
    
    def create_ego(self):
        bp=self.blueprint_library.filter("mercedes")[0]       
        spawn_point=random.choice(self.spawnpoints)
        self.ego=self.world.spawn_actor(bp, spawn_point)
        self.ego.set_autopilot(True,self.tm_port)
        self.actor_list.append(self.ego)
        cam_bp=self.blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x","620")
        cam_bp.set_attribute("image_size_y","480")
        cam_bp.set_attribute("fov","90")
        cam_sp = carla.Transform(carla.Location(x=2.5,z=0.7))
        camera=self.world.spawn_actor(cam_bp,cam_sp,attach_to=self.ego)
        self.actor_list.append(camera)
    
    def generatecars(self,blueprint_library,spawn_points,number_of_vehicles,port):
        blueprints=blueprint_library.filter("vehicle.*")
        for i in range(number_of_vehicles):
            blueprint = random.choice(blueprints)
            sp=random.choice(spawn_points)
            try:
                car=self.world.spawn_actor(blueprint, sp)
                car.set_autopilot(True,port)
                self.actor_list.append(car)
            except Exception as e:
                pass

    def calculate_speed(self,velocity_vector):
        x = velocity_vector.x
        y = velocity_vector.y
        z = velocity_vector.z

        speed = math.sqrt(x**2 + y**2 + z**2)
        
        return round(speed,0)
    
    def get_relative_position(self,ego_velocity, exo_velocity, ego_position, exo_position):
        relative_position = carla.Vector3D(
            exo_position.x - ego_position.x,
            exo_position.y - ego_position.y,
            exo_position.z - ego_position.z
        )
        
        distance = math.sqrt(
            relative_position.x ** 2 +
            relative_position.y ** 2 +
            relative_position.z ** 2
        )

        if distance != 0:
            relative_direction = carla.Vector3D(
                relative_position.x / distance,
                relative_position.y / distance,
                relative_position.z / distance
            )
        else:
            relative_direction = carla.Vector3D(0.0, 0.0, 0.0)
        
        relative_velocity = carla.Vector3D(
            exo_velocity.x - ego_velocity.x,
            exo_velocity.y - ego_velocity.y,
            exo_velocity.z - ego_velocity.z
        )
        
        dot_product = (
            relative_velocity.x * relative_position.x +
            relative_velocity.y * relative_position.y +
            relative_velocity.z * relative_position.z
        )
        movement_direction = "towards" if dot_product < 0 else "away"
        
        relative_location = ""
        if relative_position.x > 0:
            relative_location += "front"
        elif relative_position.x < 0:
            relative_location += "back"
        
        if relative_position.y > 0:
            relative_location += " right" if relative_location else "right"
        elif relative_position.y < 0:
            relative_location += " left" if relative_location else "left"
        
        relative_location = relative_location.strip()
        
    
        return relative_position, relative_direction,distance,relative_location,movement_direction

    def get_exo_vehicles_s(self, snapshot, ego_snapshot):
        """
        Extracts data about external vehicles relative to the ego vehicle.

        Parameters:
            snapshot (carla.WorldSnapshot): Snapshot of the current simulation state.
            ego_snapshot (carla.ActorSnapshot): Snapshot of the ego vehicle.

        Returns:
            list: A list of dictionaries, each representing an external vehicle and its state.
        """
        exos = []
        ego_loc = ego_snapshot.get_transform().location
        ego_vel = ego_snapshot.get_velocity()
        

        # Iterate through all actor snapshots in the simulation
        for actor_snapshot in snapshot:
            # Skip the ego vehicle
            if actor_snapshot.id == self.ego.id:
                continue

            # Get the actor's transform and velocity
            actor_transform = actor_snapshot.get_transform()
            actor_location = actor_transform.location
            actor_velocity = actor_snapshot.get_velocity()

            # Compute relative position and movement details
            relative_pos, relative_dir, distance, relative_location, movement_direction = \
                self.get_relative_position(
                    ego_loc, ego_vel, actor_location, actor_velocity
                )

            # Only consider vehicles within a 40-meter distance
            if distance < 40:
                exo_spd = self.calculate_speed(actor_velocity)
                exo = {
                    "exo_rotation_pitch": actor_transform.rotation.pitch,
                    "exo_rotation_yaw": actor_transform.rotation.yaw,
                    "exo_rotation_roll": actor_transform.rotation.roll,
                    "exo_position_x": actor_location.x,
                    "exo_position_y": actor_location.y,
                    "exo_position_z": actor_location.z,
                    "exo_velocity_x": actor_velocity.x,
                    "exo_velocity_y": actor_velocity.y,
                    "exo_velocity_z": actor_velocity.z,
                    "exo_Speed": exo_spd,
                    "exo_trafic_l_state":"None",
                    "distance_from_exo": distance,
                    "exo_relative_location": relative_location,
                    "exo_relative_movement_dir": movement_direction
                }
                exos.append(exo)

        return exos

    def get_frame_s(self, snapshot):
        if self.current_frame_count % 20 == 0:
            # Retrieve the ego vehicle's actor snapshot
            ego_snapshot = snapshot.find(self.ego.id)
            if ego_snapshot is None:
                raise ValueError("Ego vehicle not found in the snapshot.")

            # Extract ego vehicle's location, velocity, and transformation
            ego_transf = ego_snapshot.get_transform()
            ego_pos = ego_transf.location
            ego_vel = ego_snapshot.get_velocity()
            ego_spd = self.calculate_speed(ego_vel)
            
            # Get ego traffic light state (use traffic light object from the ego vehicle)
            ego_traffic_light = self.ego.get_traffic_light()
            ego_traffic_light_state = (
                ego_traffic_light.get_state() if ego_traffic_light else "None"
            )

            # Get external vehicles (exos) data
            exos = self.get_exo_vehicles_s(snapshot, ego_snapshot)

            # Update current frame data
            self.current_frame = {
                "frames": [
                    {
                        "ego_rotation_pitch": ego_transf.rotation.pitch,
                        "ego_rotation_yaw": ego_transf.rotation.yaw,
                        "ego_rotation_roll": ego_transf.rotation.roll,
                        "ego_position_x": ego_pos.x,
                        "ego_position_y": ego_pos.y,
                        "ego_position_z": ego_pos.z,
                        "ego_velocity_x": ego_vel.x,
                        "ego_velocity_y": ego_vel.y,
                        "ego_velocity_z": ego_vel.z,
                        "ego_speed": ego_spd,
                        "ego_trafic_l_state": str(ego_traffic_light_state),
                        "exo_vehicles": exos
                    }
                ]
            }

        # Increment frame counter
        self.current_frame_count += 1

    def run(self):
        self.init_carla()
        self.create_ego()
        self.generatecars(self.blueprint_library,self.spawnpoints,30,self.tm_port)
        
        
        id=self.world.on_tick(self.get_frame_s)
        while not keyboard.is_pressed('space'):
            if self.current_frame_count%20==1:
                resp=asyncio.run(self.Get_response_predict(self.url,self.current_frame))
                print(resp)
        self.cleanup()
        os._exit(0) 

    def cleanup(self):
        for actor in self.actor_list:
            actor.destroy()
        print('Process finished, actors removed.')

    async def send_json_to_fastapi(self,url: str, json_data: dict):
        """
        Sends JSON data to a FastAPI server and retrieves the response.

        Args:
            url (str): The endpoint of the FastAPI server.
            json_data (dict): The JSON payload to send.

        Returns:
            dict: The JSON response from the server.
        """
        async with httpx.AsyncClient() as client:
            try:
                # Send POST request with JSON data
                response = await client.post(url, json=json_data)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx, 5xx)
                return response.json()  # Parse and return the JSON response
            except httpx.HTTPStatusError as exc:
                print(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
                return {"error": str(exc)}
            except Exception as exc:
                print(f"An error occurred: {exc}")
                return {"error": str(exc)}

    async def send_multiple_requests(self,url: str, payloads):
        """
        Sends multiple JSON payloads to a FastAPI server concurrently.

        Args:
            url (str): The endpoint of the FastAPI server.
            payloads (list[dict]): A list of JSON payloads to send.

        Returns:
            list[dict]: A list of JSON responses from the server.
        """
        tasks = [self.send_json_to_fastapi(url, payload) for payload in payloads]
        return await asyncio.gather(*tasks)
    
    async def Get_response_predict(self,url,payload):
        resp=await self.send_json_to_fastapi(url,payload)
        return resp
  

if __name__ == "__main__":

    host_ip='localhost'
    base_port=2000
    api_url = "http://127.0.0.1:8000/items/"

    pm=Prediction_maker(api_url=api_url,host_ip=host_ip,base_port=base_port)
    pm.run()

    