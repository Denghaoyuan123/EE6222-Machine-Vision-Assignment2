import torch
from tqdm import tqdm

def inference_loop(model, model_path, data_loader, device):
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Inferencing..')
    
    model_folder = []
    model_pred = []
    model_logits = []
        
    for i, (batch_data) in pbar:
        folder = batch_data[0]
        data = batch_data[1].to(device)
        
        # Compute output & loss
        with torch.no_grad():
            logits = model.forward(data)
        
        _, pred = torch.max(logits, 1)
        
        model_folder.extend(folder)
        model_pred.extend(pred.to("cpu").numpy())

        model_logits.extend(logits.to("cpu").numpy())

    model_results = [model_folder, model_logits, model_pred]

    return model_results