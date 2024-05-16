import torch
import time
import copy
from tqdm import tqdm


def evaluate(dataloader, model, criterion, args):
    """
    This function evaluates the model on a dataloader. It computes the loss and accuracy of the model.
    The weights of the model are not updated during the evaluation. 

    Args:
        dataloader (pytorch Dataloader): Dataloader that will be used to feed the model during evaluation
        model (nn.Module): model that will be evaluated
        criterion (torch.nn): criterion that will be used to compute the loss
        args (dict): arguments of the training (device, epochs, ...)

    Returns:
        loss(float): loss of the model on the dataloader
        accuracy(float): accuracy of the model on the dataloader
        prediction(list): list of the model's predictions
    """
    loss = 0
    correct = 0
    total = 0
    prediction = []
    model.eval()
    with torch.no_grad():
        for i, (image, labels) in enumerate(dataloader):
            # distribution of image and labels to all GPUs
            image = image.to(args['device'], non_blocking=True)
            labels = labels.to(args['device'], non_blocking=True)
            labels = labels.float()
            outputs = model(image)
            outputs =  outputs.squeeze(1)
            loss_ = criterion(outputs,labels)

            loss += loss_
            total += outputs.size(0)
            # print(predicted)
            # print(labels)
            # print(predicted.size())
            # print(labels.size())
            correct += (outputs == labels).sum().item()
            prediction.append(outputs)
                        
    model.train()
    loss = (loss/total).item()
    accuracy = (correct/total)*100
    return loss, accuracy, prediction

def train(train_loader, val_loader, model, optimizer, criterion, args, scheduler,verbose = 0):
    """
    This function trains the model on the train_loader and evaluates it on the val_loader at each epoch.

    Args:
        train_loader (pytorch Dataloader): Dataloader that will be used to feed the model during training
        val_loader (pytorch Dataloader): Dataloader that will be used to feed the model during evaluation
        model (nn.Module): model to train
        optimizer (torch.optim): optimizer that will be used to update the model's weights (Adam, SGD...)
        criterion (torch.nn): criterion that will be used to compute the loss (CrossEntropy, MSELoss...)
        args (dict): arguments of the training (device, epochs, batch size, patience...)
        scheduler (_type_): scheduler that will be used to update the learning rate
        verbose (int, optional): set to 1 to print the loss and accuracy at each epoch. Defaults to 1. 

    Returns:
            'best_model_state'
            'train_losses'
            'train_accuracies'
            'val_losses'
            'val_accuracies'
            'duration'
    """
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    time_start = time.time()
    
    best_val_loss = float('inf')
    consecutive_no_improvement = 0
    tot_epoch = args['epochs']
    patience = args['patience']
    
    if args['AMP']== True:
        scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(args['epochs']):
        loop = tqdm(train_loader)
        train_loss = 0
        train_correct = 0
        train_total = 0
        for i, (image, labels) in enumerate(loop):
            # distribution of image and labels to all GPUs
            image = image.to(args['device'], non_blocking=True)
            labels = labels.to(args['device'], non_blocking=True)
            labels = labels.float()
            if args['AMP'] == True:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(image)
                    outputs = outputs.squeeze(1)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                predictions = (outputs > 0.5).float()
                train_loss += loss
                train_total += outputs.size(0)
                train_correct += (predictions == labels).sum().item()
            
            else:
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(image)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels)
                # Backward pass
                loss.backward()
                # Optimize
                optimizer.step()
                
                predictions = (outputs > 0.5).float()
                train_loss += loss
                train_total += outputs.size(0)
                train_correct += (predictions == labels).sum().item()
                
            if scheduler is not None:
                scheduler.step()
            
            loop.set_description(f"Epoch [{epoch+1}/{args['epochs']}]")
            loop.set_postfix(loss=(train_loss/train_total).item(), acc=(train_correct/train_total)*100)

        # Evaluate at the end of the epoch on the train set
        train_loss = (train_loss/train_total).item()
        train_accuracy = (train_correct/train_total)*100
        if verbose == 1:
            print("\t Train loss : ", train_loss, "& Train accuracy : ", train_accuracy)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate at the end of the epoch on the val set
        val_loss, val_accuracy,_ = evaluate(val_loader, model, criterion, args)
        if verbose == 1:
            print("\t Validation loss : ", val_loss, "& Validation accuracy : ", val_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
    
        # Early stopping check
        if val_loss < best_val_loss:
            if verbose == 1:
                print(f"Validation loss improved from {best_val_loss} to {val_loss}. Saving model.")
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1

        if consecutive_no_improvement >= patience:
            print(f"Early stopping after {consecutive_no_improvement} epochs of no improvement.")
            tot_epoch = epoch
            break

    duration = time.time() - time_start
    print('Finished Training in:', duration, 'seconds with mean epoch duration:', duration/tot_epoch, ' seconds')
    results = {'best_model_state':best_model,
               'train_losses': train_losses,
               'train_accuracies': train_accuracies,
               'val_losses': val_losses,
               'val_accuracies': val_accuracies,
               'duration':duration}
    return results



    
