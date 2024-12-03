from fastapi import FastAPI
from api_request_types.request_models import InputFormat
from models.neural_networks.working.dataset_processing import DatasetProcessing
from api_request_types.model import Predicter
import json
import os

app = FastAPI()
data_processing = DatasetProcessing()
model = Predicter(
    os.path.join(os.getcwd(), "models", "neural_networks", "working", "trained.pth")
)


@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}


@app.post("/items/")
def read_item(input_data: InputFormat):
    graph_list = data_processing.create_graph_from_api(input_data)
    predicted_class, predicted_bool = model.predict(graph_list)

    class_number = predicted_class.item()
    lane_change = bool(predicted_bool[0, 0].item())
    turn = bool(predicted_bool[0, 1].item())
    result_dict = {
        "predicted_class": class_number,
        "lane_change": lane_change,
        "turn": turn,
    }
    a = json.dumps(result_dict)
    return a
