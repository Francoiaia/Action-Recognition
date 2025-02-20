# -*- coding: utf-8 -*-
from datetime import datetime

import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
from tqdm.auto import tqdm
from src.model import build_model
from image_datasets import get_data_loaders, shuffle_csv, CustomDataset
from utils import save_model, save_plots, SaveBestModel
from class_names import class_names

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to train our network for")
parser.add_argument(
    "-lr",
    "--learning-rate",
    type=float,
    dest="learning_rate",
    default=0.001,
    help="Learning rate for training the model",
)
parser.add_argument("-b", "--batch-size", dest="batch_size", default=32, type=int)
parser.add_argument(
    "-ft", "--fine-tune", dest="fine_tune", action="store_true", help="pass this to fine tune all layers"
)
parser.add_argument("--save-name", dest="save_name", default="model", help="file name of the final model to save")
parser.add_argument("--scheduler", action="store_true", help="use learning rate scheduler if passed")
args = parser.parse_args()


# Training function.
def train(model, trainloader, optimizer, criterion):
    model.train()
    print("Training")
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    prog_bar = tqdm(trainloader, total=len(trainloader), bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")
    for i, data in enumerate(prog_bar):
        counter += 1
        image, labels = data["image"], data["label"]
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation.
        loss.backward()
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100.0 * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# Validation function.
def validate(model, testloader, criterion):
    model.eval()
    print("Validation")
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    prog_bar = tqdm(testloader, total=len(testloader), bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")
    with torch.no_grad():
        for i, data in enumerate(prog_bar):
            counter += 1

            image, labels = data["image"], data["label"]
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100.0 * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        """
        patience (int): Number of epochs to wait before stopping
        min_delta (float): Minimum change in monitored value to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


if __name__ == "__main__":
    early_stopping = EarlyStopping(patience=10)
    start_time = datetime.now()
    batch_size = 64
    lr = 0.001
    epochs = 30
    is_scheduler = True
    # Create a directory with the model name for outputs.
    out_dir = os.path.join("../..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    # Load the training and validation datasets.
    train_df, valid_df = shuffle_csv()
    dataset_train = CustomDataset(train_df, class_names, is_train=True)
    dataset_valid = CustomDataset(valid_df, class_names, is_train=False)
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Classes: {class_names}")
    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid, batch_size=batch_size)
    # Learning_parameters.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")
    # Load the model.
    model = build_model(fine_tune=True, num_classes=len(class_names)).to(device)
    print(model)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    # Optimizer.
    # optimizer = optim.SGD(
    # model.parameters(), lr=lr, momentum=0.9, nesterov=True
    # )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
    # Loss function.
    criterion = nn.CrossEntropyLoss()
    # Initialize `SaveBestModel` class.
    save_best_model = SaveBestModel()
    # LR scheduler.
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7], gamma=0.1)
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader, criterion)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        save_best_model(valid_epoch_loss, epoch, model, out_dir, args.save_name)
        if is_scheduler:
            scheduler.step()
        early_stopping(valid_epoch_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        print("-" * 50)
    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion, out_dir, args.save_name)
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, out_dir)
    print("TRAINING COMPLETE")
    end_time = datetime.now()
    delta = end_time - start_time
    minutes = delta.total_seconds() / 60
    print(f"Time elapsed: {minutes:.2f} minutes")
