import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import torch
from sklearn.metrics import accuracy_score, top_k_accuracy_score

def train_loop(model, dataloader, device, optimizer, accum_iter, 
               neptune_run=None, scheduler=None, epoch_info=''):
    
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    model.train()
    losses = []
    lr_list = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    for i, batch_data in pbar:
        #print(i,'i')
        folder = batch_data[0]
        data = batch_data[1].to(device)
        target = batch_data[2].to(device)
        #print('data',data,'target',target)

        with autocast():
            logits = model.forward(data)
            loss = criterion(logits, target)
            
        #rmse = torch.sqrt(mse_loss(logits.sigmoid() * 100, target * 100))

        if scheduler is not None:
            lr = scheduler.get_lr()[0]
        else:
            lr = optimizer.param_groups[0]['lr']

        if neptune_run is not None:
            neptune_run[f'{epoch_info.split(",")[0]} LR'].log(lr)

        pbar.set_description(
            f'>> {epoch_info} - Train Loss : {loss.item():4.4f} LR : {lr:.1e}')
        losses.append(loss.item())
        
        # Backward pass
        loss = loss / accum_iter
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()
        
        if ((i + 1) % accum_iter == 0) or (i + 1 == len(pbar)):
            scaler.step(optimizer)

            if scheduler:
                lr_list.append(scheduler.get_lr()[0])
                scheduler.step()
            
            scaler.update()
            optimizer.zero_grad()
        print(lr_list,'lr_list')
    return lr_list, losses 

def val_loop(model, dataloader, device, label_list, k=5,
             epoch_info=''):

    CELoss = nn.CrossEntropyLoss()

    model.eval()
    losses = []
    avg_score = []
    avg_k_score = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    for i, batch_data in pbar:
        
        folder = batch_data[0]
        data = batch_data[1].to(device)
        target = batch_data[2].to(device)

        # Compute output & loss
        with torch.no_grad():
            logits = model.forward(data)
            loss = CELoss(logits, target)
        
        _, pred = torch.max(logits, 1)

        score = accuracy_score(target.to("cpu"), pred.to("cpu"))
        top_k_score = top_k_accuracy_score(target.to("cpu"), logits.to("cpu"), k=k, labels=label_list)

        pbar.set_description(
            f'>> {epoch_info} - Val Loss   : {loss.item():4.4f}, Model accuracy score   : {score:4.4f}')

        losses.append(loss.item())
        avg_score.append(score)
        avg_k_score.append(top_k_score)

    return losses, avg_score, avg_k_score