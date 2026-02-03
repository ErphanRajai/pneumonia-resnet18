import torch
from src import data_setup, model_builder, engine
import torch.optim as optim

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, test_loader, class_names = data_setup.create_dataloaders("data/train", "data/test")
model = model_builder.create_resnet18_model().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Loop
for epoch in range(10):
    train_loss = engine.train_step(model, train_loader, criterion, optimizer, device)
    test_loss = engine.test_step(model, test_loader, criterion, device)
    print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} | Test Loss {test_loss:.4f}")

torch.save(model.state_dict(), "models/pneumonia_model.pth")