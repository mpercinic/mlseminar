import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
from SRToolkit.evaluation import ParameterEstimator
from scipy.optimize import minimize
from load_dataset import datasets
from parsers import *


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_hidden_layers, output_size=1):
        super(SimpleNN, self).__init__()

        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.output_size = output_size

        layers = []
        current_size = input_size

        # Build hidden layers (each with half the nodes of previous layer) # TODO: fix?
        for i in range(num_hidden_layers):
            next_size = max(1, current_size // 2)  # Ensure at least 1 node
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.ReLU())
            current_size = next_size

        # Output layer
        layers.append(nn.Linear(current_size, output_size))

        self.network = nn.Sequential(*layers)

        self.double()

    def forward(self, x):
        return self.network(x)

    def get_architecture(self):
        """Print the network architecture"""
        print(f"\n{'=' * 50}")
        print(f"Regression Neural Network Architecture")
        print(f"{'=' * 50}")
        print(f"Input size: {self.input_size}")
        print(f"Number of hidden layers: {self.num_hidden_layers}")

        current_size = self.input_size
        for i in range(self.num_hidden_layers):
            next_size = max(1, current_size // 2)
            print(f"Hidden Layer {i + 1}: {current_size} -> {next_size} (ReLU)")
            current_size = next_size

        print(f"Output Layer: {current_size} -> {self.output_size}")
        print(f"{'=' * 50}\n")


class RegressionDataset(Dataset):
    """Simple dataset wrapper for regression data"""

    def __init__(self, X, y):
        self.X = torch.DoubleTensor(X)
        self.y = torch.DoubleTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(model, train_loader, criterion, optimizer, num_epochs=100, verbose=True):
    """
    Train the regression model

    Args:
        model: The neural network model
        train_loader: DataLoader with training data
        criterion: Loss function
        optimizer: Optimization algorithm
        num_epochs: Number of training epochs
        verbose: Whether to print training progress
    """
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1) if batch_y.dim() == 1 else batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if verbose and (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')


def evaluate_model(model, test_loader, criterion):
    """
    Evaluate the model on test data

    Args:
        model: The neural network model
        test_loader: DataLoader with test data
        criterion: Loss function

    Returns:
        Average test loss
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1) if batch_y.dim() == 1 else batch_y)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    return avg_loss

np.random.seed(42)

num_epochs = 2000
train_test_split = 0.7

first = True
for dataset in datasets:
    '''if first:
        first = False
        continue'''
    # data preparation
    n_const = dataset["num_constants"]
    c_min, c_max = dataset["kwargs"]["constant_range"]
    X = np.transpose(dataset["X"])
    x_train, x_test = X[:, :int(train_test_split*X.shape[1])], X[:, int(train_test_split*X.shape[1]):]
    y = np.transpose(dataset["y"])
    y_train, y_test = y[:int(train_test_split*len(y))], y[int(train_test_split*len(y)):]
    c0_1 = np.random.rand(n_const)
    cs = [c_min + (c_max - c_min) * c0_1[i] for i in range(n_const)]
    #cs = [torch.tensor(c_min + (c_max - c_min) * c0_1[i], requires_grad=True) for i in range(n_const)]
    exec(expr_to_code(dataset["expression"]))
    while np.isnan(rhs).any():
        c0_1 = np.random.rand(n_const)
        cs = [c_min + (c_max - c_min) * c0_1[i] for i in range(n_const)]
        exec(expr_to_code(dataset["expression"]))
    print(cs)


    print(dataset["ground_truth"])
    #print(dataset["expression"])

    print("Scipy minimize results:")
    exec(expr_to_scipy(dataset["expression"]))
    print(res.message)
    print("Values of the parameters: " + str(res.x))
    print("Number of evaluations: " + str(res.nfev))
    print("Error: " + str(res.fun))
    exec(scipy_test_evaluation(dataset["expression"]))
    print("Test error: " + str(np.mean((rhs - y_test)**2)))
    print()

    # comp graphs just comparing rhs
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)

    cs1 = [torch.tensor(c, requires_grad=True) for c in cs]
    optimizer = optim.Adam(cs1, lr=0.001)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        exec(expr_to_tensor(dataset["expression"], 'cs1'))
        loss = torch.mean((predicted - y_train) ** 2)
        loss.backward()
        optimizer.step()
    print("Computational graph results:")
    print("Values of the parameters: " + str([c.item() for c in cs]))
    print("Values of the parameters: " + str([c.item() for c in cs1]))
    print("Number of evaluations: " + str(num_epochs))
    print("Error: " + str(loss.item()))

    # TODO: add evaluation for test data
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test)
    exec(torch_test_evaluation(dataset["expression"]))
    print("Test error: " + str(torch.mean((predicted - y_test)**2).item()))

    #comp graphs for rhs and parameters
    # can't use parameters because that is what we are searching -> not a standard machine learning task

    '''cs2 = [torch.tensor(c, requires_grad=True) for c in cs]
    constants = torch.tensor(dataset["constants"], requires_grad=True)
    optimizer = optim.Adam(cs2, lr=0.001)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        exec(expr_to_tensor(dataset["expression"], 'cs2'))
        se = torch.stack([(cs2[i] - dataset["constants"][i]) ** 2 for i in range(len(cs2))])
        loss = torch.mean(se)
        loss.backward()
        optimizer.step()
    print("Computational graph results:")
    print("Values of the parameters: " + str([c.item() for c in cs]))
    print("Values of the parameters: " + str([c.item() for c in cs2]))
    print("Number of evaluations: " + str(num_epochs))
    print("Error: " + str(torch.mean((predicted - y_train) ** 2).item()))'''

    # SimpleNN
    train_dataset = RegressionDataset(torch.transpose(x_train, 0, 1), y_train)
    test_dataset = RegressionDataset(torch.transpose(x_test, 0, 1), y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create model with 10 input nodes and 3 hidden layers
    # Architecture will be: 10 -> 5 -> 2 -> 1 (output always 1 node)
    model = SimpleNN(input_size=X.shape[0], num_hidden_layers=1)
    model.get_architecture()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Training the model...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs, verbose=True)

    # Evaluate the model
    test_loss = evaluate_model(model, test_loader, criterion)
    print(f'\nTest Loss: {test_loss:.4f}')

    # Make a prediction on a single sample
    sample_input = torch.FloatTensor(x_test[0:1])
    prediction = model(sample_input)
    print(f'\nSample Prediction: {prediction.item():.4f}')
    print(f'Actual Value: {y_test[0]:.4f}')
    input()



# TODO: variable parameters: # of instances per equation, # of epochs, starting values for parameters

x1_list = [random.randint(0, 1000) for _ in range(1000)]
x2_list = [random.randint(0, 1000) for _ in range(1000)]
y_list  = [62.4*x1_list[i]**2 - 73.7*x2_list[i] for i in range(1000)]

x1 = torch.tensor(x1_list)
x2 = torch.tensor(x2_list)
y = torch.tensor(y_list)


c = np.random.rand(2)
c1 = torch.tensor([(10.0- 1e-8)*c[0]-5], requires_grad=True)
c2 = torch.tensor([(10.0- 1e-8)*c[1]-5], requires_grad=True)

res = minimize(lambda x: np.mean((x[0] * np.array(x1_list) ** 2 - x[1] * np.array(x2_list) - np.array(y_list))**2), [(10.0- 1e-8)*c[0]-5, (10.0- 1e-8)*c[1]-5])

#optimizer = optim.Adam([c1, c2], lr=0.1)
optimizer = optim.AdamW([c1, c2], lr=0.01)

num_epochs = 20000
for epoch in range(num_epochs):
    optimizer.zero_grad()

    predicted = c1 * x1 ** 2 - c2 * x2
    print(c1.item(), c2.item())

    loss = torch.mean((predicted - y) ** 2)
    loss.backward()
    optimizer.step()

print(c1.item(), c2.item())

X = np.stack((np.array(x1_list), np.array(x2_list)), axis=1)
pe = ParameterEstimator(X, np.array(y_list), seed=42)

# Estimate the parameters of the expression
print(pe.estimate_parameters(["C", "*", "(", "X_0", ")", "^2", "-", "C", "*", "X_1"]))