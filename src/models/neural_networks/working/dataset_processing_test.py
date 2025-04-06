import torch
import math
# PyTorch Geometric imports
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
import json
from api_request_types.request_models import InputFormat
import os

# For debugging or general utilities (optional)

class DatasetProcessing():
    def __init__(self):
        print(f"WORKING DIRECTORY::::{os.getcwd()}")

        try:
            with open("mapper.json","r") as f:
                self.mapper_d=json.load(f)
        except:
            with open(os.path.join(os.getcwd(),"models","neural_networks","working","mapper.json"),"r") as f:
                self.mapper_d=json.load(f)
        
    
    def create_graph_array_basic(self,train_d: dict) -> list:
        graph_list = []
        for k, v in train_d.items():
            nodes = torch.tensor([], dtype=torch.float)
            edge_indexes = torch.tensor([[], []], dtype=torch.long)
            edge_features = torch.tensor([], dtype=torch.float)
            k = str(k)
            v = dict(v)
            all_vehicle = []

            ego = v.pop("ego_vehicle")
            ego_pos = ego["position"]
            ego_wayp = ego["waypoint_location"]
            ego_node = torch.tensor(
                [
                    [
                        ego_pos["x"],
                        ego_pos["y"],
                        ego_pos["z"],
                        ego_wayp["x"],
                        ego_wayp["y"],
                        ego_wayp["z"],
                        ego["speed"],
                        0,
                        0,
                        0,
                    ]
                ],
                dtype=torch.float,
            )
            nodes = torch.cat((nodes, ego_node), dim=0)
            all_vehicle.append({
                    "index": 0,
                    "pos": ego_pos,
                    "dir": {"x": 0, "y": 0, "z": 0},
                    "node_data": ego_node
                    })

            for idx, (key, value) in enumerate(v.items(), start=1):
                exo_pos = value["Position"]
                exo_rotation = value["Rotation"]
                exo_velocity = value["Velocity"]
                exo_rel_pos = value["relative_position"][0]
                exo_rel_dir = value["relative_direction"]
                exo_node = torch.tensor(
                    [
                        [
                            exo_pos["x"],
                            exo_pos["y"],
                            exo_pos["z"],
                            exo_rotation["pitch"],
                            exo_rotation["yaw"],
                            exo_rotation["roll"],
                            value["Speed"],
                            exo_velocity["x"],
                            exo_velocity["y"],
                            exo_velocity["z"],
                        ]
                    ],
                    dtype=torch.float,
                )
                nodes = torch.cat((nodes, exo_node), dim=0)
                all_vehicle.append({
                    "index": idx,
                    "pos": exo_pos,
                    "dir": exo_rel_dir,
                    "node_data": exo_node
                    })
                #e_index = torch.tensor([[0, index], [index, 0]], dtype=torch.long)
                #edge_indexes = torch.cat((edge_indexes, e_index), dim=1)
                num_nodes = len(all_vehicle):
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:
                            pos_i = all_vehicle[i]["pos"]
                            pos_j = all_vehicle[j]["pos"]

                            dx = pos_i["x"] - pos_j["x"]
                            dx = pos_i["y"] - pos_j["y"]
                            dx = pos_i["z"] - pos_j["z"]
                            distance = math.sqrt(dx**2 + dy**2 + dz**2)

                            if distance < 30.0:
                                edge_indexes = torch.cat(
                                    (edge_indexes, torch.tensor([[i], [j]], dtype=torch.long)), dim=1
                                )

                                rel_pos = {
                                    "x": pos_j["x"] - pos_i["x"],
                                    "y": pos_j["y"] - pos_i["y"],
                                    "z": pos_j["z"] - pos_i["z"],
                                }
                                rel_dir = all_vehicle[j]["dir"]
                                edge_attr = torch.tensor(
                                    [[
                                        rel_pos["x"],
                                        rel_pos["y"],
                                        rel_pos["z"],
                                        rel_dir["x"],
                                        rel_dir["y"],
                                        rel_dir["z"],
                                    ]],
                                    dtype=torch.float,
                                )
                                edge_features = torch.cat((edge_features, e_attr), dim=0)


                # num_nodes = nodes.size(0)
                # edges = torch.cartesian_prod(torch.arange(num_nodes), torch.arange(num_nodes))
                # edges_no_self = edges[edges[:, 0] != edges[:, 1]]
                # edge_attributes = torch.ones(edges_no_self.size(0), dtype=torch.float)
                # index += 1
                # graph = Data(x=nodes, edge_index=edges_no_self, edge_attr=edge_attributes)


                graph = Data(x=nodes, edge_index=edge_indexes, edge_attr=edge_features)
                graph_list.append(graph)

    return graph_list
    

    def modified(self,dict_list: list):
        # label list is a list of tuple(nextdir, [lanechange,turn])
        graph_list = []
        label_list=[]
        for train_d in dict_list:
            for k, v in train_d.items():
                nodes = torch.tensor([], dtype=torch.float)
                edge_indexes = torch.tensor([[], []], dtype=torch.long)
                edge_features = torch.tensor([], dtype=torch.float)
                k = str(k)
                v = dict(v)
                ego = v.pop("vehicle EGO",False)
                ego_rotation=ego["Rotation"]
                ego_position=ego["Position"]
                ego_velocity=ego["Velocity"]
                ego_trafic_l_state=self.mapper_d["Traffic_Light_State"][ego["Traffic_Light_State"]]

                ego_node = torch.tensor(
                    [
                        [
                            ego_rotation["pitch"],
                            ego_rotation["yaw"],
                            ego_rotation["roll"],
                            ego_position["x"],
                            ego_position["y"],
                            ego_position["z"],
                            ego_velocity["x"],
                            ego_velocity["y"],
                            ego_velocity["z"],
                            ego["speed"],
                            ego_trafic_l_state,
                            0,
                            0
                        ]
                    ],
                    dtype=torch.float,
                )
                nodes = torch.cat((nodes, ego_node), dim=0)
                
                all_agents = [{
                    "index": 0,
                    "pos": ego_position,
                    "dir": {"x": 0, "y": 0, "z": 0},
                    "node_data": ego_node,
                    "is_ego": True
                }]

                for idx, (key, value) in enumerate(v.items(), start=1):
                    exo_rotation=value["Rotation"]
                    exo_position=value["relative_position"][0]
                    exo_velocity=value["Velocity"]
                    exo_relative_location=self.mapper_d["relative_location"][value["relative_location"]]
                    exo_relative_movement_dir=self.mapper_d["relative_movement_direction"][value["relative_movement_direction"]]
                    exo_trafic_l_state=self.mapper_d["Traffic_Light_State"][value["Traffic_Light_State"]]


                    exo_node = torch.tensor(
                        [
                            [
                                exo_rotation["pitch"],
                                exo_rotation["yaw"],
                                exo_rotation["roll"],
                                exo_position["x"],
                                exo_position["y"],
                                exo_position["z"],
                                exo_velocity["x"],
                                exo_velocity["y"],
                                exo_velocity["z"],
                                value["Speed"],
                                exo_trafic_l_state,
                                exo_relative_location,
                                exo_relative_movement_dir
                            ]
                        ],
                        dtype=torch.float,
                    )
                    nodes = torch.cat((nodes, exo_node), dim=0)
                    all_agents.append({
                        "index": idx,
                        "pos": exo_position,
                        "dir":  {"x": 0, "y": 0, "z": 0},,
                        "node_data": exo_node
                        "is_ego": False
                        "label": [value["Lanechange"], value["Turn"]] #placeholder
                    })

                    num_nodes = len(all_agents)
                    for i in range(num_nodes):
                        for j in range(num_nodes):
                            if i != j:
                                pos_i = all_agents[i]["pos"]
                                pos_j = all_agents[j]["pos"]

                                dx = pos_i["x"] - pos_j["x"]
                                dy = pos_i["y"] - pos_j["y"]
                                dz = pos_i["z"] - pos_j["z"]
                                distance = math.sqrt(dx**2 + dy**2 + dz**2)
                                if distance < 30.0:
                                        edge_indexes = torch.cat(
                                           (edge_indexes, torch.tensor([[i], [j]], dtype=torch.long)), dim=1
                                        )
                                        edge_attr = torch.tensor([[distance]], dtype=torch.float)
                                        edge_features = torch.cat((edge_features, edge_attr), dim=0)
                                        edge_features = torch.cat((edge_features, edge_attr), dim=0)
                
                node_labels = []
                for agent in all_agent:
                    if agent.get("is_ego", False):
                        node_labels.append(None)
                    else:
                        label = [int(agent["label"][0]), int(agent["label"][1])]
                        node_labels.append(label)
                # # TEST------------------------------
                # num_nodes = nodes.size(0)
                # edges = torch.cartesian_prod(torch.arange(num_nodes), torch.arange(num_nodes))
                # edges_no_self = edges[edges[:, 0] != edges[:, 1]]
                # edges_reshaped = edges_no_self.t()
                # edge_attributes = torch.ones(edges_no_self.size(0), dtype=torch.float)
                # index += 1
                # graph = Data(x=nodes, edge_index=edges_reshaped, edge_attr=edge_attributes)
                # # TEST-------------------------

                graph = Data(x=nodes, edge_index=edge_indexes, edge_attr=edge_features)
                transform=NormalizeFeatures()
                graph=transform(graph)
                graph_list.append(graph)
                label_list.append(node_labels)

    return graph_list, label_list
    
    def create_graph_from_api(self, frames:InputFormat):
        graph_list=[]

        for frame in frames.frames:
            nodes = torch.tensor([], dtype=torch.float)
            edge_indexes = torch.tensor([[], []], dtype=torch.long)
            edge_features = torch.tensor([], dtype=torch.float)
            ego_node = torch.tensor(
                [
                    [
                        frame.ego_rotation_pitch,
                        frame.ego_rotation_yaw,
                        frame.ego_rotation_roll,
                        frame.ego_position_x,
                        frame.ego_position_y,
                        frame.ego_position_z,
                        frame.ego_velocity_x,
                        frame.ego_velocity_y,
                        frame.ego_velocity_z,
                        frame.ego_speed,
                        self.mapper_d["Traffic_Light_State"][frame.ego_trafic_l_state],
                        0,
                        0
                    ]
                ],
                dtype=torch.float,
            )
            nodes = torch.cat((nodes, ego_node), dim=0)
            all_agents = [{
                "index": 0,
                "pos": {
                    "x": frame.ego_position_x,
                    "y": frame.ego_position_y,
                    "z": frame.ego_position_z,
                },
                "node_data": ego_node
            }]
            for idx, exo in enumerate(frame.exo_vehicles, start=1):
                exo_node = torch.tensor(
                    [
                        [
                            exo.exo_rotation_pitch,
                            exo.exo_rotation_yaw,
                            exo.exo_rotation_roll,
                            exo.exo_position_x,
                            exo.exo_position_y,
                            exo.exo_position_z,
                            exo.exo_velocity_x,
                            exo.exo_velocity_y,
                            exo.exo_velocity_z,
                            exo.exo_Speed,
                            self.mapper_d["Traffic_Light_State"][exo.exo_trafic_l_state],
                            self.mapper_d["relative_location"][exo.exo_relative_location],
                            self.mapper_d["relative_movement_direction"][exo.exo_relative_movement_dir]
                        ]
                    ],
                    dtype=torch.float,
                )
                nodes = torch.cat((nodes, exo_node), dim=0)
                
                all_agents.append({
                    "index": idx,
                    "pos": {
                        "x": exo.exo_position_x,
                        "y": exo.exo_position_y,
                        "z": exo.exo_position_z
                    },
                    "node_data": exo_node
                })
            num_nodes = len(all_agents)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        pos_i = all_agents[i]["pos"]
                        pos_j = all_agents[j]["pos"]
    
                        dx = pos_i["x"] - pos_j["x"]
                        dy = pos_i["y"] - pos_j["y"]
                        dz = pos_i["z"] - pos_j["z"]
                        distance = math.sqrt(dx**2 + dy**2 + dz**2)
                         if distance < 30.0:
                                            edge_indexes = torch.cat(
                                               (edge_indexes, torch.tensor([[i], [j]], dtype=torch.long)), dim=1
                                            )
                                            edge_attr = torch.tensor([[distance]], dtype=torch.float)
                                            edge_features = torch.cat((edge_features, edge_attr), dim=0)
                                            edge_features = torch.cat((edge_features, edge_attr), dim=0)
    
                    
            graph = Data(x=nodes, edge_index=edge_indexes, edge_attr=edge_features)
            graph_list.append(graph)
        return graph_list
    
    def create_sequences(self,graph_array,sequence_length):
        sequences = [graph_array[i:i + sequence_length] for i in range(0, len(graph_array), sequence_length)]
        return sequences
    
    def create_sequence_and_labels(self,graph_array,label_list,sequence_length):
        sequences = [graph_array[i:i + sequence_length] for i in range(0, len(graph_array), sequence_length)]
        labels=label_list[::sequence_length]
        return sequences,labels
