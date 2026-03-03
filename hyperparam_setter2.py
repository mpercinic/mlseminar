import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
from SRToolkit.evaluation import ParameterEstimator
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
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
        '''print(f"\n{'=' * 50}")
        print(f"Regression Neural Network Architecture")
        print(f"{'=' * 50}")
        print(f"Input size: {self.input_size}")
        print(f"Number of hidden layers: {self.num_hidden_layers}")'''

        current_size = self.input_size
        for i in range(self.num_hidden_layers):
            next_size = max(1, current_size // 2)
            #print(f"Hidden Layer {i + 1}: {current_size} -> {next_size} (ReLU)")
            current_size = next_size

        #print(f"Output Layer: {current_size} -> {self.output_size}")
        #print(f"{'=' * 50}\n")


class RegressionDataset(Dataset):
    """Simple dataset wrapper for regression data"""

    def __init__(self, X, y):
        self.X = torch.DoubleTensor(X)
        self.y = torch.DoubleTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(model, train_loader, criterion, optimizer, num_epochs=100, verbose=False):
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
            total_loss += loss

    avg_loss = total_loss / len(test_loader)
    return avg_loss

def run_optimization(method, bounds):
    #print("Scipy minimize results:")
    #print(expr_to_scipy(dataset["expression"], method, bounds))
    '''res = minimize(lambda c: np.mean(
        ((x_train[2] + x_train[1]) / (c[0] + (x_train[2] * x_train[1]) / (x_train[0] ** 2)) - y_train) ** 2),
                   [c.item() for c in cs], method='Nelder-Mead', bounds=[(-5, 5) for _ in range(n_const)])'''
    '''print(cs)
    print(expr_to_scipy(dataset["expression"], method, bounds))'''
    exec(expr_to_scipy(dataset["expression"], method, bounds), globals())
    #print(res.message)
    '''print("Values of the parameters: " + str(res.x))
    print("Number of evaluations: " + str(res.nfev))
    print("Error: " + str(res.fun))'''
    exec(scipy_test_evaluation(dataset["expression"]), globals())
    '''print(rhs)
    print("Test error: " + str(np.mean((rhs - y_test) ** 2)))
    print()'''
    #print(np.isnan(rhs).any())
    return np.sqrt(mean_squared_error(y_test, rhs)), mean_absolute_error(y_test, rhs), r2_score(y_test, rhs)

'''def run_comp_graph(x_train, x_test, y_train, y_test, cs):
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)

    cs1 = [torch.tensor(c, requires_grad=True) for c in cs]
    optimizer = optim.Adam(cs1, lr=0.001)
    for epoch in range(num_epochs_cg):
        optimizer.zero_grad()
        print(expr_to_tensor(dataset["expression"]))
        exec('predicted = (x_train[2] + x_train[1]) / (cs1[0] + (x_train[2] * x_train[1]) / (x_train[0] ** 2))', globals(), {'cs1': cs1})
        #exec(expr_to_tensor(dataset["expression"]), globals(), {'cs1': cs1})
        loss = torch.mean((predicted - y_train) ** 2)
        loss.backward()
        optimizer.step()
    #print("Computational graph results:")
    #print("Values of the parameters: " + str([c.item() for c in cs]))
    #print("Values of the parameters: " + str([c.item() for c in cs1]))
    #print("Number of evaluations: " + str(num_epochs_cg))
    #print("Error: " + str(loss.item()))

    # TODO: add evaluation for test data
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test)
    exec(torch_test_evaluation(dataset["expression"]), globals())
    return np.sqrt(mean_squared_error(y_test, predicted)), mean_absolute_error(y_test, predicted), r2_score(y_test, predicted)
'''
def run_nn():
    # SimpleNN
    train_dataset = RegressionDataset(x_train, y_train)
    test_dataset = RegressionDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create model with 10 input nodes and 3 hidden layers
    # Architecture will be: 10 -> 5 -> 2 -> 1 (output always 1 node)
    model = SimpleNN(input_size=x_train.shape[1], num_hidden_layers=max([int(x_train.shape[1]/2), 1]))
    model.get_architecture()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    #print("Training the model...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs_nn)

    # Evaluate the model
    rmse_loss = np.sqrt(evaluate_model(model, test_loader, mean_squared_error))
    mae_loss = evaluate_model(model, test_loader, mean_absolute_error)
    r2_loss = evaluate_model(model, test_loader, r2_score)

    return rmse_loss, mae_loss, r2_loss

def run_rand_forest():
    '''grid_search = GridSearchCV(RandomForestRegressor(max_features='log2'), param_grid=rand_forest_params, cv=5)
    grid_search.fit(x_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    print("Best Estimator:", grid_search.best_estimator_)
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values(by=["rank_test_score"])
    results_df = results_df.set_index(
        results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
    ).rename_axis("kernel")
    print(results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]].to_string())'''

    rf_model = RandomForestRegressor(max_features='log2', n_estimators=50)
    rf_model.fit(x_train, y_train)
    y_pred = rf_model.predict(x_test)
    return np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)

def run_knn():
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn_model.fit(x_train_scaled, y_train)
    y_pred = knn_model.predict(x_test_scaled)
    return np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)

np.random.seed(42)

num_epochs_nn = 1000
num_epochs_cg = 5000
test_size = 0.3

rand_forest_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 5]
}

knn_params = {
    'n_neighbors': [1, 2, 5, 10, 15],
    'weights': ['distance', 'uniform'],
}

dt_params = {
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 5]
}

svr_params = {
    'kernel': ['linear', 'rbf'],
    'C': [2, 5],
    'epsilon': [0.005, 0.01]
}
# TODO: best: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}

first = True

for dataset in datasets:
    # data preparation
    n_const = dataset["num_constants"]
    c_min, c_max = dataset["kwargs"]["constant_range"]
    '''c0_1 = np.random.rand(n_const)
    cs = [c_min + (c_max - c_min) * c0_1[i] for i in range(n_const)]'''
    #cs = [torch.tensor(c_min + (c_max - c_min) * c0_1[i], requires_grad=True) for i in range(n_const)]
    print(dataset["ground_truth"])
    '''exec(expr_to_code(dataset["expression"]))
    while np.isnan(rhs).any():
        c0_1 = np.random.rand(n_const)
        cs = [c_min + (c_max - c_min) * c0_1[i] for i in range(n_const)]
        exec(expr_to_code(dataset["expression"]))'''

    # print(dataset["expression"])

    grid_search = GridSearchCV(SVR(), param_grid=svr_params, cv=5)
    grid_search.fit(dataset["X"], dataset["y"])
    #print("Best Parameters:", grid_search.best_params_)
    #print("Best Estimator:", grid_search.best_estimator_)
    results_df = pd.DataFrame(grid_search.cv_results_)
    #results_df = results_df.sort_values(by=["rank_test_score"])
    results_df = results_df.set_index(
        results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
    ).rename_axis("kernel")
    #print(results_df[["params", "mean_test_score"]].to_string())
    if first:
        means = results_df["mean_test_score"].to_numpy()
        kernels = results_df["params"].to_numpy()
        first = False
    else:
        means = means + results_df["mean_test_score"].to_numpy()

sort = np.argsort(means)
kernel_best = kernels[sort[-1]]
print(kernel_best)
