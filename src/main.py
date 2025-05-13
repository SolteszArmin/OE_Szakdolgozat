from fastapi import FastAPI
from api_request_types.request_models import InputDataFormat
from models.neural_networks.working.dataset_processing import DatasetProcessing
from api_request_types.model import Predicter
import json
import os

app = FastAPI()
data_processing = DatasetProcessing()
model = Predicter(
    os.path.join(os.getcwd(), "models", "neural_networks", "working", "trained_model.pth")
)


@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}


@app.post("/items/")
def read_item(input_data: InputDataFormat):
    input_data_dict = input_data.model_dump()
    predicted_next_move, predicted_speed_change, returned_node_ids = model.predict(input_data_dict)

    result_dict = {
        "predicted_next_action": predicted_next_move,
        "predicted_speed_change": predicted_speed_change,
        "ids": returned_node_ids,
    }
    a = json.dumps(result_dict)
    return a
