from pydantic import BaseModel, EmailStr


class ExoVehicle(BaseModel):
    exo_rotation_pitch: float
    exo_rotation_yaw: float
    exo_rotation_roll: float
    exo_position_x: float
    exo_position_y: float
    exo_position_z: float
    exo_velocity_x: float
    exo_velocity_y: float
    exo_velocity_z: float
    exo_Speed: float
    distance_from_exo: float
    exo_trafic_l_state: str
    exo_relative_location: str
    exo_relative_movement_dir: str


class FrameFormat(BaseModel):
    ego_rotation_pitch: float
    ego_rotation_yaw: float
    ego_rotation_roll: float
    ego_position_x: float
    ego_position_y: float
    ego_position_z: float
    ego_velocity_x: float
    ego_velocity_y: float
    ego_velocity_z: float
    ego_speed: float
    ego_trafic_l_state: str
    exo_vehicles: list[ExoVehicle]

class InputFormat(BaseModel):
    frames: list[FrameFormat]
