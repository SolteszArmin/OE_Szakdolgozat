import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class liner(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(liner,self).__init__()
        self.asd=nn.Linear(input_dim,output_dim)
        
    def forward(self,x):
        return self.asd(x)
    

# Define the model with two linear layers
class TwoStageLinearRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoStageLinearRegression, self).__init__()
        # First linear regression layer (stage 1)
        self.linear1 = nn.Linear(input_size, hidden_size)
        # Second linear regression layer (stage 2)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Stage 1: first linear transformation
        x = self.linear1(x)
        # Stage 2: second linear transformation with the output from stage 1
        x = self.linear2(x)
        return x

# Create dummy dataset where y = 2 * (3 * x) = 6 * x
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32).to(device)  # Inputs moved to GPU
y = torch.tensor([[6.0], [12.0], [18.0], [24.0]], dtype=torch.float32).to(device) # Labels moved to GPU

# Initialize the composite model and move it to the GPU
input_size = 1   # Input is a single feature
hidden_size = 1  # Output of first layer is also a single value (intermediate output)
output_size = 1  # Final output is a single value (predicted y)

model = TwoStageLinearRegression(input_size, hidden_size, output_size).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Train the model for a fixed number of epochs
num_epochs = 100

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 100 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model with new input data (move input to GPU)
test_input = torch.tensor([[5.0]], dtype=torch.float32).to(device)

# Prediction using the trained model (move prediction back to CPU to print)
predicted_output = model(test_input).item()
print(f'Prediction from two-stage linear regression model: {predicted_output:.2f}')

test_inputs = torch.linspace(0, 6, steps=100).view(-1, 1).to(device)  # Generate inputs from 0 to 6

# Prediction using the trained model (move prediction back to CPU to plot)
with torch.no_grad():
    predicted_outputs = model(test_inputs).cpu()  # Move predictions to CPU for plotting

# Plot the original data and the predicted line
plt.figure(figsize=(8,6))

# Plot original data points (in red)
plt.scatter(X.cpu(), y.cpu(), color="red", label="Original data")

# Plot predicted data points (in blue)
plt.plot(test_inputs.cpu(), predicted_outputs, color="blue", label="Model predictions")

# Labels and title
plt.title("Two-Stage Linear Regression Model: y = 6 * x")
plt.xlabel("Input X")
plt.ylabel("Output Y")
plt.legend()

# Show plot
plt.grid(True)
plt.show()