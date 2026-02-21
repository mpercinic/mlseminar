def expr_to_scipy(expr, method, bounds):
    code = 'res = minimize(lambda c: np.mean((' # TODO: add root?
    const_counter = 0
    for c in expr:
        if c == 'sqrt': code += 'np.sqrt'
        elif c == 'sin': code += 'np.sin'
        elif c == 'cos': code += 'np.cos'
        elif c == 'exp': code += 'np.exp'
        elif c == 'arcsin': code += 'np.arcsin'
        elif c == 'tanh': code += 'np.tanh'
        elif c == 'ln': code += 'np.log'
        elif c == '^2': code += '**2'
        elif c == '^3': code += '**3'
        elif c == 'u-': code += '-'
        elif c[:2] == 'C_':
            code += 'c[' + str(const_counter) + ']'
            const_counter += 1
        elif c[:2] == 'X_':
            code += 'x_train[' + str(c[2:]) + ']'
        else: code += c
    code += '-y_train)**2), [c.item() for c in cs], method=' + method
    if bounds is not None: code += ', bounds=' + bounds
    return code + ')'

def expr_to_tensor(expr):
    code = 'predicted = '
    const_counter = 0
    for c in expr:
        if c == 'sqrt': code += 'torch.sqrt'
        elif c == 'sin': code += 'torch.sin'
        elif c == 'cos': code += 'torch.cos'
        elif c == 'exp': code += 'torch.exp'
        elif c == 'arcsin': code += 'torch.arcsin'
        elif c == 'tanh': code += 'torch.tanh'
        elif c == 'ln': code += 'torch.log'
        elif c == '^2': code += '**2'
        elif c == '^3': code += '**3'
        elif c == 'u-': code += 'torch.neg'
        elif c[:2] == 'C_':
            code += 'cs1[' + str(const_counter) + ']'
            const_counter += 1
        elif c[:2] == 'X_':
            code += 'x_train_tensor[' + str(c[2:]) + ']'
        else: code += c
    return code

def expr_to_code(expr):
    code = 'rhs = '
    const_counter = 0
    for c in expr:
        if c == 'sqrt': code += 'np.sqrt'
        elif c == 'sin': code += 'np.sin'
        elif c == 'cos': code += 'np.cos'
        elif c == 'exp': code += 'np.exp'
        elif c == 'arcsin': code += 'np.arcsin'
        elif c == 'tanh': code += 'np.tanh'
        elif c == 'ln': code += 'np.log'
        elif c == '^2': code += '**2'
        elif c == '^3': code += '**3'
        elif c == 'u-': code += '-'
        elif c[:2] == 'C_':
            code += 'cs[' + str(const_counter) + '].item()'
            const_counter += 1
        elif c[:2] == 'X_':
            code += 'np.transpose(dataset["X"])[' + str(c[2:]) + ']'
        else: code += c
    return code

def scipy_test_evaluation(expr):
    code = 'rhs = '
    const_counter = 0
    for c in expr:
        if c == 'sqrt': code += 'np.sqrt'
        elif c == 'sin': code += 'np.sin'
        elif c == 'cos': code += 'np.cos'
        elif c == 'exp': code += 'np.exp'
        elif c == 'arcsin': code += 'np.arcsin'
        elif c == 'tanh': code += 'np.tanh'
        elif c == 'ln': code += 'np.log'
        elif c == '^2': code += '**2'
        elif c == '^3': code += '**3'
        elif c == 'u-': code += '-'
        elif c[:2] == 'C_':
            code += 'res.x[' + str(const_counter) + ']'
            const_counter += 1
        elif c[:2] == 'X_':
            code += 'x_test[' + str(c[2:]) + ']'
        else: code += c
    return code

def torch_test_evaluation(expr):
    code = 'prediction = '
    const_counter = 0
    for c in expr:
        if c == 'sqrt': code += 'torch.sqrt'
        elif c == 'sin': code += 'torch.sin'
        elif c == 'cos': code += 'torch.cos'
        elif c == 'exp': code += 'torch.exp'
        elif c == 'arcsin': code += 'torch.arcsin'
        elif c == 'tanh': code += 'torch.tanh'
        elif c == 'ln': code += 'torch.log'
        elif c == '^2': code += '**2'
        elif c == '^3': code += '**3'
        elif c == 'u-': code += 'torch.neg'
        elif c[:2] == 'C_':
            code += 'cs1[' + str(const_counter) + ']'
            const_counter += 1
        elif c[:2] == 'X_':
            code += 'x_test_tensor[' + str(c[2:]) + ']'
        else: code += c
    return code

