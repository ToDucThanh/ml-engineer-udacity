import os
import torch
import json

from net import Net

JSON_CONTENT_TYPE = "application/json"

def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(device)
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.model.load_state_dict(torch.load(f))
    
    model.eval()
    
    return model
    
def input_fn(request_body, request_content_type):
    if request_content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(request_body)
        
        return input_data

    raise Exception("Requested unsupported ContentType in Accept: " + request_content_type)
        

def predict_fn(input_data, model):
    with torch.no_grad():
        predictions = model(input_data)
    
    return predictions

def output_fn(predictions, content_type):
    if content_type == JSON_CONTENT_TYPE:
        res = predictions.cpu().detach().numpy().tolist()
        
        return json.dumps(res)

    raise Exception("Requested unsupported ContentType in Accept: " + content_type)
    

    
    