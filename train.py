import numpy as np
import time
import torch
import torchvision

# Main training function

def train(model, loader, mask_loader, optimizer, device):
    loss_list = []

    for epoch in range(loader):
        print('Starting training....{}/{}'.format(epoch+1, loader))
        loss_sub_list = []
        start = time.time()
        for images, targets in mask_loader:
            images = list(image.to(device) for image in images)
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
            
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            loss_sub_list.append(loss_value)
            
            # update optimizer and learning rate
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            #lr_scheduler.step()
        end = time.time()
            
        #print the loss of epoch
        epoch_loss = np.mean(loss_sub_list)
        loss_list.append(epoch_loss)
        print('Epoch loss: {:.3f} , time used: ({:.1f}s)'.format(epoch_loss, end-start))