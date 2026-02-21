import numpy as np
from SRToolkit.dataset import SR_benchmark

# Create the Feynman benchmark suite
feynman = SR_benchmark.feynman("./data/feynman")
datasets = feynman.datasets


for dataset_name in datasets:
    dataset = feynman.create_dataset(dataset_name)
    expression = []
    num_constants = 0
    constants = []
    for s in datasets[dataset_name]['ground_truth']:
        if s == 'pi':
            expression.append('C_' + str(num_constants))
            num_constants += 1
            constants.append(np.pi)
        elif s not in datasets[dataset_name]['symbol_library'].symbols and s not in ['(', ')']:
            expression.append('C_' + str(num_constants))
            num_constants += 1
            constants.append(float(s))
        else:
            expression.append(s)
    if datasets[dataset_name]['ground_truth'] == ['exp', '(', 'u-', 'X_0', '^2', '/', '2', ')', '/', 'sqrt', '(', '2',
                                                  '*', 'pi', ')']:
        expression = ['exp', '(', 'u-', '(', 'X_0', '^2', ')', '/', 'C_0', ')', '/', 'sqrt', '(', 'C_1', '*', 'C_2', ')']
    elif datasets[dataset_name]['ground_truth'] == ['(', 'u-', 'X_0', '*', '(', 'X_1', '^2', '*', 'X_1', '^2', ')', '/', '(', '(', '2', '*', '(', '4', '*', 'pi', '*', 'X_4', ')', '^2', ')', '*', '(', 'X_2', '/', '(', '2', '*', 'pi', ')', ')', '^2', ')', '*', '(', '1', '/', 'X_3', '^2', ')', ')']:
        expression = ['(', 'u-', '(', 'X_0', ')', '*', '(', 'X_1', '^2', '*', 'X_1', '^2', ')', '/', '(', '(', 'C_0', '*', '(', 'C_1', '*', 'C_2', '*', 'X_4', ')', '^2', ')', '*', '(', 'X_2', '/', '(', 'C_3', '*', 'C_4', ')', ')', '^2', ')', '*', '(', 'C_5', '/', 'X_3', '^2', ')', ')']
    datasets[dataset_name]['expression'] = expression
    datasets[dataset_name]['num_constants'] = num_constants
    datasets[dataset_name]['constants'] = constants
    datasets[dataset_name]['dataset_name'] = dataset_name
    datasets[dataset_name]['X'] = dataset.X
    #noise = np.random.uniform(0.95, 1.05, size=dataset.X.shape)
    #datasets[dataset_name]['X'] = dataset.X * noise
    datasets[dataset_name]['y'] = dataset.y
    #noise = np.random.uniform(0.95, 1.05, size=dataset.y.shape)
    #datasets[dataset_name]['y'] = dataset.y * noise

datasets = [datasets[dataset_name] for dataset_name in datasets if datasets[dataset_name]['num_constants'] > 0]
print("Number of equations with free parameters: " + str(len(datasets)) + '/100')

