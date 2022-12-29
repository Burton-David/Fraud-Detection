import torch
import torch.nn as nn

# Define the model


class FraudDetectionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# Instantiate the model
model = FraudDetectionModel(input_size=30, hidden_size=20, output_size=1)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(100):
    # Generate some fake data
    fake_data = torch.randn(32, 30)
    fake_labels = torch.rand(32, 1) > 0.5

    # Reset the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(fake_data)
    loss = criterion(outputs, fake_labels)

    # Backward pass
    loss.backward()

    # Update the weights
    optimizer.step()

    # Print the loss
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Test the model
test_data = torch.randn(16, 30)
test_labels = torch.rand(16, 1) > 0.5
predictions = model(test_data)

# Print the accuracy
accuracy = (predictions.round() == test_labels).float().mean()
print(f"Accuracy: {accuracy:.4f}")
