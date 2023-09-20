import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X, Y, theta, alpha, iterations):
    m = Y.shape[0] #     number of examples
    cost=[]
    for i in range(iterations):
        gradient = (np.subtract(np.dot(np.dot(X.T, X),theta), np.dot(X.T,Y))) /m
        theta = np.subtract(theta, alpha * gradient) # simultaneously update theta when each iteration finished
        iter_cost = np.dot(np.subtract(np.dot(X,theta), Y).T, np.subtract(np.dot(X,theta),Y))/2/m
        cost.append(iter_cost)
        if  i == 0 or i == iterations - 1:
            print('iteration#', i,'\ncost:',iter_cost,'\ngradient:',gradient,'\ntheta:',theta,'\n')
    return theta, cost

# get data with arbitrary number of features and one target
def get_data(filename, file_delimiter):
    data = np.genfromtxt(filename, delimiter=file_delimiter) # read data into (97,2) np array
    X0 = np.ones(data.shape[0]) # X0=1 for all examples by convention
    X1_Xn = data[:, :data.shape[1] - 1] #X1 to Xn
    Y = data[:,data.shape[1]-1]
#     feature scaling for multiple features
    if X1_Xn.shape[1] >= 2:
        X1_Xn, stdX, avgX = feature_scaling(X1_Xn)
        X = np.column_stack((X0,X1_Xn)) # make X the training set (97,2), each row is one example
        return X, Y, stdX, avgX
    else:
        X = np.column_stack((X0,X1_Xn))
        return X, Y

# scale a matrix of features by (x - mean(x))/standard_deviation(x)
def feature_scaling(original):
    std = np.std(original,axis = 0)
    avg = np.average(original, axis=0)
#     array in the shape of original array, filled with standard deviation of each column
    std_deviation = np.full_like(original,std)
#     array in the shape of original array, filled with average of each column
    average = np.full_like(original, avg)

    scaled = (original - average) / std_deviation
    return scaled, std, avg

# plot the model and data only for models with one feature
def plot_model(X1, Y, theta):
    plt.figure(1, figsize = (8,6))
    plt.title('model')
    plt.xlabel('Population of city in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.scatter(X1, Y, marker = 'x') # scatter plot for example data
    x_axis = np.linspace(min(X1)-1, max(Y)+1, 100)
    print('theta = ', theta,'\nmodel: h(theta)= ', theta[0],' + ', theta[1], ' * x1')
    plt.plot(x_axis, theta[0] + theta[1] * x_axis, c= "red") # linear model
    plt.show()

# plot cost function with respect to gradient descent iterations
def plot_cost(cost):
    plt.figure(0, figsize = (8,6))
    plt.title('cost function')
    plt.xlabel('number of gradient descent iterations')
    plt.ylabel('cost')
    plt.plot(np.linspace(1, len(cost), len(cost)), cost, c= "red")
    plt.show()

