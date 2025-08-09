import torch
import numpy as np
import joblib as jlb
from tqdm import tqdm
from typing import Tuple
from pathlib import Path
from torch import FloatTensor
from torch.utils.data import DataLoader

from data.data_classes import CircuitDataset


def train(model, 
          criterion, 
          optimizer, 
          data_dir: Path, 
          num_epochs: int, 
          batch_size: int, 
          train_outdir: Path) -> Tuple[Path, Path]:
    """
    Run the training
    """
    # Use CUDA if available
    device = "cpu"
    model.to(device)

    num_workers = 2

    # Datasets
    train_dataset = CircuitDataset(mode="train", data_dir=data_dir)
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True,
                              num_workers=num_workers)

    val_dataset = CircuitDataset(mode="val", data_dir=data_dir)
    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            num_workers=num_workers)

    # Lists to store all cumulative losses
    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(num_epochs)):
        # Train the model
        model.train()
        train_loss = 0

        for inputs, labels in train_loader:
            # Move data to device
            inputs.to(device)
            labels.to(device)

            # Zero the gradients from the previous iteration
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Compute gradients
            loss.backward()

            # Update model's parameters
            optimizer.step()

            # Add to the training loss
            train_loss += loss.item()

        # Store the epoch's averaged train loss
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move to device
                inputs.to(device)
                labels.to(device)
                
                # Get predictions
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Store the epoch's averaged validation loss
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Save the model and losses
    train_outdir.mkdir(parents=True, exist_ok=True)

    weights_path = train_outdir.joinpath(f"{model.__class__.__name__}_weights.pt")
    torch.save(model.state_dict(), weights_path)

    losses_path = train_outdir.joinpath(f"{model.__class__.__name__}_train_losses.npz")
    np.savez(losses_path, train_losses=np.array(train_losses), val_losses=np.array(val_losses))

    return weights_path, losses_path


def test(model, criterion, weights_path: Path, data_dir: Path, test_outdir: Path, batch_size: int) -> Path:
    """
    Test the model
    """
    # Use CUDA if available
    device = "cpu"
    model.to(device)
    
    num_workers = 2

    # Dataset
    test_dataset = CircuitDataset(mode="test", data_dir=data_dir)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    # Evaluate the model and gather predictions
    model.eval()
    
    test_loss = 0
    test_preds = []
    test_labels = []

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in test_loader:
            # Move to device
            inputs.to(device)
            labels.to(device)
            
            # Perform a forward pass
            outputs = model(inputs)
            
            # Compute loss and add it
            test_loss += criterion(outputs, labels).item()
            
            # Store predictions and labels
            test_preds.extend(outputs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            
    # Average the loss over the number of batches
    test_loss = test_loss / len(test_loader)
    print(f"Test loss: {test_loss:.4f}")
    
    # Save predictions
    test_outdir.mkdir(parents=True, exist_ok=True)  # Should have been created during training

    preds_path = test_outdir.joinpath(f"{model.__class__.__name__}_test_predictions.npz")
    np.savez(preds_path, predicted=np.array(test_preds), observed=np.array(test_labels))

    return preds_path


def dist_eval(model, target_dist: FloatTensor, scaler_path: Path) -> list:
    """
    """
    # Use CUDA if available
    device = "cpu"
    model.to(device)

    # Scale the target distribution samples as we did for the training data
    scaler = jlb.load(scaler_path)
    scaled_obs = scaler.transform(target_dist.reshape((1, -1)))

    # Move to device
    inputs = torch.from_numpy(scaled_obs).float()
    inputs.to(device)

    # Get model's predictions
    model.eval()

    with torch.no_grad():
        outputs = model(inputs)

    return outputs.flatten().tolist()
