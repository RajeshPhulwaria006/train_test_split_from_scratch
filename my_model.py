import numpy as np

# function to implement train_test_split from scratch using numpy.
def train_test_split(x, y, test_size=0.2, shuffle = True, random_state=None):
    X = np.array([x])
    Y = np.array([y])
    
    if random_state is not None:
        np.random.seed(random_state)

    # shuffle the data 
    if shuffle:
        indices = np.random.permutation(X.shape[0])
    else:
        indices = np.arange(X.shape[0])
        
    # split the size of data for training
    count = int(len(X) * test_size)
    train = indices[count:]     # split data according to test size in %
    test = indices[:count]
    
    X_test, X_train = X[test], X[train]
    Y_test, Y_train = Y[test], Y[train]
    
    return X_train, X_test, Y_train, Y_test

x = np.arange(20).reshape(10, 2)
y = np.arange(10)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print("X_train:\n", x_train)
print("X_test:\n", x_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test)