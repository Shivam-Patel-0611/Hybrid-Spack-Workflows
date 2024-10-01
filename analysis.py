import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Load preprocessed data
data = np.load('data.npy')
labels = np.load('labels.npy')

# Convert data to PyTorch tensors and move to GPU
data = torch.FloatTensor(data).cuda()
labels = torch.tensor(labels, dtype=torch.int64).cuda()

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), 'simple_nn.pth')
