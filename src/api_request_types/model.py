from models.neural_networks.working.GNN import GATv2SequenceModel
from torch_geometric.data import Batch
import torch


class Predicter():
    def __init__(self, weights_path: str) -> None:
        model = GATv2SequenceModel(13, 16, 1, 4, 2)
        model.load_state_dict(torch.load(weights_path))
        model.eval()
        self.model = model

    def predict(self, data):
        batched_graph = Batch.from_data_list(data)
        # Step 2: Extract the graph data
        x = batched_graph.x  # Node features
        edge_index = batched_graph.edge_index  # Edge indices
        edge_attr = batched_graph.edge_attr  # Edge attributes
        batch = (
            batched_graph.batch
        )  # Batch vector (indicates which graph each node belongs to)

        # Step 3: Sequence length (number of graphs in this sequence)
        sequence_length = [
            len(data)
        ]  # Only one sequence, so its length is the full sequence

        # Step 4: Pass through the model
        out, out_bool = self.model(x, edge_index, edge_attr, batch, sequence_length)
        predicted_class = torch.argmax(out)
        probabilities = torch.sigmoid(out_bool)
        threshold = 0.5
        bool_values = probabilities > threshold
        return predicted_class, bool_values
