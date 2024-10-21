import torch
import torch.nn as nn
import torch.optim as optim
from bitnet import BitNetModel
from torchvision import datasets, transforms

# Define the model
class BitNetClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(BitNetClassifier, self).__init__()
        self.bitnet = BitNetModel(input_dim, hidden_dim, num_classes, num_layers=3)

    def forward(self, x):
        return self.bitnet(x.view(x.size(0), -1))

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize the model
model = BitNetClassifier(784, 256, 10).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Evaluation
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
print(f'\nTest set: Average loss: {test_loss:.4f}, '
      f'Accuracy: {correct}/{len(test_loader.dataset)} '
      f'({100. * correct / len(test_loader.dataset):.2f}%)')