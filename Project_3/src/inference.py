import os
import torch
import json

from net import Net

JSON_CONTENT_TYPE = "application/json"

def model_fn(model_dir):
    print("Loading model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(device)
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    
    model.eval()
    
    return model
    
def input_fn(request_body, request_content_type):
    print("Input1")
    if request_content_type == "application/json":
        print("Input2")
        input_data = json.loads(request_body)
        
        return input_data

    else:
        raise Exception("Requested unsupported ContentType in Accept: " + content_type)
        
        return

def predict_fn(input_data, model):

    print("Got input Data: {}".format(input_data))
    
    with torch.no_grad():
        predictions = model(input_data)
    
    return predictions

def output_fn(predictions, content_type):
    print("PREDICTION: ", predictions)
    if content_type == "application/json":
        res = predictions.cpu().detach().numpy().tolist()
        
        return json.dumps(res)

    raise Exception("Requested unsupported ContentType in Accept: " + content_type)
    

    
    