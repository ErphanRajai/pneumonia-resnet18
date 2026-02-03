import torch
from tqdm.notebook import tqdm

def train_step(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(dataloader)

def test_step(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    return test_loss / len(dataloader)