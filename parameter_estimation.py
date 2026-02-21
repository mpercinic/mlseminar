import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import time
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from load_dataset import datasets
from parsers import *



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
    return np.sqrt(mean_squared_error(y_test, rhs)), mean_absolute_error(y_test, rhs), r2_score(y_test, rhs), mean_absolute_percentage_error(y_test, rhs)


def run_rand_forest():

    rf_model = RandomForestRegressor(max_features='log2', n_estimators=200)
    rf_model.fit(x_train, y_train)
    y_pred = rf_model.predict(x_test)
    print(mean_absolute_percentage_error(y_test, y_pred))
    return np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred), mean_absolute_percentage_error(y_test, y_pred)

def run_knn():
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn_model.fit(x_train_scaled, y_train)
    y_pred = knn_model.predict(x_test_scaled)
    print(mean_absolute_percentage_error(y_test, y_pred))
    return np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred), mean_absolute_percentage_error(y_test, y_pred)

np.random.seed(42)

num_epochs_nn = 1000
num_epochs_cg = 5000
test_size = 0.3

rand_forest_params = {
    'n_estimators': [20, 50, 100, 150, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'bootstrap': [True, False]
}

invalids = [0] * 6
results_final = [[], [], [], [], [], []]
results_time = [[], [], [], [], [], []]

for dataset in datasets:
    # data preparation
    n_const = dataset["num_constants"]
    c_min, c_max = dataset["kwargs"]["constant_range"]
    c0_1 = np.random.rand(n_const)
    cs = [c_min + (c_max - c_min) * c0_1[i] for i in range(n_const)]
    #cs = [torch.tensor(c_min + (c_max - c_min) * c0_1[i], requires_grad=True) for i in range(n_const)]
    print(dataset["ground_truth"])
    exec(expr_to_code(dataset["expression"]))
    while np.isnan(rhs).any():
        c0_1 = np.random.rand(n_const)
        cs = [c_min + (c_max - c_min) * c0_1[i] for i in range(n_const)]
        exec(expr_to_code(dataset["expression"]))

    # print(dataset["expression"])

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    rf_results = []
    knn_results = []
    nm_results = []
    lbfgsb_results = []
    slsqp_results = []
    comp_graph_results = []
    nn_results = []

    x_train, x_test, y_train, y_test = train_test_split(dataset["X"], dataset["y"], test_size=0.3, random_state=42, shuffle=True)

    start = time.time()
    rf_results.append(run_rand_forest())
    end = time.time()
    results_time[0].append(end - start)
    start = time.time()
    knn_results.append(run_knn())
    end = time.time()
    results_time[1].append(end - start)
    #nn_results.append(run_nn())
    x_train, x_test = np.transpose(x_train), np.transpose(x_test)

    nmmin = [100000, 100000, -1]
    lbfgsbmin = [100000, 100000, -1]
    slsqpmin = [100000, 100000, -1]
    cgmin = [100000, 100000, -1]

    for counter in range(5):

        start = time.time()
        nmres = run_optimization("'Nelder-Mead'", '[(-5, 5) for _ in range(n_const)]')
        if nmres[0] < nmmin[0]: nmmin = nmres
        end = time.time()
        results_time[2].append(end - start)
        start = time.time()
        lbfgsbres = run_optimization("'L-BFGS-B'", '[(-5, 5) for _ in range(n_const)]')
        if lbfgsbres[0] < lbfgsbmin[0]: lbfgsbmin = lbfgsbres
        end = time.time()
        results_time[3].append(end - start)
        start = time.time()
        slsqpres = run_optimization("'SLSQP'", '[(-5, 5) for _ in range(n_const)]')
        if slsqpres[0] < slsqpmin[0]: slsqpmin = slsqpres
        end = time.time()
        results_time[4].append(end - start)
        #comp_graph_results.append(run_comp_graph(x_train, x_test, y_train, y_test, cs))

        x_train_tensor = torch.tensor(x_train)
        y_train_tensor = torch.tensor(y_train)

        cs1 = [torch.tensor(c, requires_grad=True) for c in cs]
        optimizer = optim.Adam(cs1, lr=0.001)
        start = time.time()
        for epoch in range(num_epochs_cg):
            optimizer.zero_grad()
            #exec('predicted = (x_train[2] + x_train[1]) / (cs1[0] + (x_train[2] * x_train[1]) / (x_train[0] ** 2))', globals())
            exec(expr_to_tensor(dataset["expression"]), globals())
            loss = torch.sqrt(torch.mean((predicted - y_train_tensor) ** 2))
            if loss < 1e-5: break
            #print(loss.item())
            loss.backward()
            optimizer.step()
        # print("Computational graph results:")
        # print("Values of the parameters: " + str([c.item() for c in cs]))
        # print("Values of the parameters: " + str([c.item() for c in cs1]))
        # print("Number of evaluations: " + str(num_epochs_cg))
        # print("Error: " + str(loss.item()))

        # TODO: add evaluation for test data
        x_test_tensor = torch.tensor(x_test)
        y_test_tensor = torch.tensor(y_test)
        exec(torch_test_evaluation(dataset["expression"]), globals())
        try:
            if np.sqrt(mean_squared_error(y_test_tensor, prediction.detach())) < cgmin[0]:
                cgmin = (np.sqrt(mean_squared_error(y_test_tensor, prediction.detach())), mean_absolute_error(y_test, prediction.detach()), r2_score(y_test, prediction.detach()), mean_absolute_percentage_error(y_test, prediction.detach()))
            #comp_graph_results.append((np.sqrt(mean_squared_error(y_test, prediction.detach())), mean_absolute_error(y_test, prediction.detach()), r2_score(y_test, prediction.detach()), mean_absolute_percentage_error(y_test, prediction.detach())))
            end = time.time()
            results_time[5].append(end - start)
        except:
            testvar = 2
            #invalids[-1] += 1
            #break

        c0_1 = np.random.rand(n_const)
        cs = [c_min + (c_max - c_min) * c0_1[i] for i in range(n_const)]
        # cs = [torch.tensor(c_min + (c_max - c_min) * c0_1[i], requires_grad=True) for i in range(n_const)]
        print(dataset["ground_truth"])
        exec(expr_to_code(dataset["expression"]))
        while np.isnan(rhs).any():
            c0_1 = np.random.rand(n_const)
            cs = [c_min + (c_max - c_min) * c0_1[i] for i in range(n_const)]
            exec(expr_to_code(dataset["expression"]))


        '''run_optimization()
        run_comp_graph(x_train, x_test, y_train, y_test)
        run_nn()'''
    nm_results.append(nmmin)
    lbfgsb_results.append(lbfgsbmin)
    slsqp_results.append(slsqpmin)
    comp_graph_results.append(cgmin)

    i = 0
    for res in [rf_results, knn_results, nm_results, lbfgsb_results, slsqp_results, comp_graph_results]:
        if len(res) == 0:
            continue
        res = np.mean(res, axis=0)
        if res[2] < 0: invalids[i] += 1
        else:
            results_final[i].append(res)
            print(f"RMSE: {res[0]:.3f}, MAE: {res[1]:.3f}, R2: {res[2]:.3f}, MAPE: {res[3]:.3f}")
        i += 1
    #input()
results_final = [np.mean(np.array(res), axis=0) for res in results_final]
print(results_final)
print(invalids)
print([np.mean(np.array(res), axis=0) for res in results_time])
