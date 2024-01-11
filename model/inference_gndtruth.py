import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np

def inference_loop(model, model_path, data_loader, device):
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    
    CELoss = nn.CrossEntropyLoss()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Inferencing..')
    
    model_folder = []
    model_pred = []
    model_logits = []
    losses = []
    model_target = []

    for i, (batch_data) in pbar:
        print(f'length of batch data: {len(batch_data)}')
        folder = batch_data[0]
        data = batch_data[1].to(device)
        target = batch_data[2].to(device)

        # Compute output & loss
        with torch.no_grad():
            logits = model.forward(data)
            loss = CELoss(logits, target)
        
        _, pred = torch.max(logits, 1)
        
        model_folder.extend(folder)
        model_pred.extend(pred.to("cpu").numpy())

        model_logits.extend(logits.to("cpu").numpy())
        losses.append(loss.item())

        model_target.extend(target.to("cpu").numpy())
        model_loss = np.mean(losses)  

    model_results = [model_folder, model_logits, model_pred, model_target, model_loss]
    print("model results",model_results)
    return model_results