from models.neural_networks.working.neural.GNN import TemporalGraphModel
from torch_geometric.data import Batch
from models.neural_networks.working.dataset_processing_test import DatasetProcessing
import torch
import json
import os


class Predicter:
    def __init__(self, weights_path: str) -> None:
        model = TemporalGraphModel(13, 64, 128, 6)
        model.load_state_dict(torch.load(weights_path))
        model.eval()
        mapper_path = os.path.join(
            os.getcwd(), "models", "neural_networks", "working", "mapper.json"
        )
        mapper_d = json.load(open(mapper_path))
        data_processing = DatasetProcessing()
        self.model = model
        self.data_processing = data_processing
        self.mapper_d = mapper_d

    def predict(self, data: dict):
        graph_list = self.data_processing.create_graphs_and_labels([data])
        graphs_sequence, _ = self.data_processing.create_sequence_and_labels(
            graph_list[0], graph_list[1], 5
        )

        manuver_change, speed_change = self.model(graphs_sequence[0])
        predicted_manover = []
        predicted_speed_change = []
        for i, j in zip(manuver_change, speed_change):
            predicted_manover.append(torch.argmax(i))
            predicted_speed_change.append(torch.argmax(j))

        returned_manover = []
        returned_speed_change = []
        returned_node_ids = graphs_sequence[0][-1].node_ids[1:]
        for pm, ps in zip(predicted_manover, predicted_speed_change):
            for k, v in self.mapper_d["next_action"].items():
                if v == pm:
                    returned_manover.append(k)
                    break
            for k, v in self.mapper_d["speed_change"].items():
                if v == ps:
                    returned_speed_change.append(k)
                    break
        return returned_manover, returned_speed_change, returned_node_ids
