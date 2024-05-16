import numpy as np
import os
import torch.optim as optim
import torch
import torchvision
from torchvision import transforms
from models.resnet18 import resnet18

from utils.DL_utils import  evaluate, train
from utils.plot_utils import plot_training, cf_matrix, clf_report
from utils.config_utils import read_config, read_cli


__VERSION__ = """
RELEASE
=======
    Name    : workflow/classic_training.py
    Release : 0.0.2
    
"""

"""
------------------------------------------------------------------------------------------------------------------------
                                                     Workflow
------------------------------------------------------------------------------------------------------------------------
"""


# Read the configuration file
# config_path = 'config/config_basics.yml'
cli = read_cli(__VERSION__)
config = read_config(cli.config_file)

# Import data
print("Importing data...")
train_dataset = torchvision.datasets.PCAM(root=os.getcwd(), split= 'train', download=True,transform=transforms.ToTensor())
val_dataset = torchvision.datasets.PCAM(root=os.getcwd(), split= 'val', download=True,transform=transforms.ToTensor())
test_dataset = torchvision.datasets.PCAM(root=os.getcwd(), split= 'test', download=True,transform=transforms.ToTensor())


# Create DataLoaders
print("Creating DataLoaders...")

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=config["training"]['batch_size'],
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=config["training"]['num_workers'],
                                           pin_memory=True,
                                           persistent_workers=True,
                                           prefetch_factor=2)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=config["training"]['batch_size'],
                                           shuffle=False,
                                           drop_last=True,
                                           num_workers=config["training"]['num_workers'],
                                           pin_memory=True,
                                           persistent_workers=True,
                                           prefetch_factor=2)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=config["training"]['batch_size'],
                                           shuffle=False,
                                           drop_last=True,
                                           num_workers=config["training"]['num_workers'],
                                           pin_memory=True,
                                           persistent_workers=True,
                                           prefetch_factor=2)

# Create model
print("Creating model...")

if config["model"]["name"] == 'resnet-18':
    model = resnet18(pretrained=config["model"]["pretrained"])
    model = model.to(config["training"]["device"])

if config["model"]["name"] == 'resnet-34':
    #TODO Add resnet34 implementation
    pass

# Define loss and optimizer
#criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config["training"]['learning_rate'])

# Train model
print("Training model...")
results = train(train_loader, val_loader, model, optimizer, criterion, config["training"], None, verbose = 1)

# Get best model
model.load_state_dict(results['best_model_state']) 

print("Evaluating model...")
loss, accuracy, prediction = evaluate(test_loader, model, criterion, config["training"])
print("Test loss: ", loss)
print("Test accuracy: ", accuracy)

# Plot training
plot_training(results, save_path= os.path.join(os.getcwd(), 'fig'), name= config["model"]['name']+ '_training.png')

# Confusion matrix
y_pred = torch.cat(prediction,dim=0)
y_pred = np.array([tensor.cpu().numpy() for tensor in y_pred])

for _,  in enumerate(test_dataset):
    y_test = y_test[1]
    y_test = np.array([tensor.cpu().numpy() for tensor in y_test])


cf_matrix(y_test, y_pred, save_path= os.path.join(os.getcwd(), 'fig'), name= config["model"]['name']+'_cf_matrix.png')

# Classification report
clf_report(y_test, y_pred, save_path= os.path.join(os.getcwd(), 'fig'), name= config["model"]['name']+'_clf_report.txt')