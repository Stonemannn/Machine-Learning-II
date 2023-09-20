import numpy as np
from function import gradient_descent, get_data, plot_model, plot_cost


# Press the green button in the gutter to run the script.
def main():
    # perform gradient descent with arbitrary number of features
    # also prints the cost for the first and last iterations for comparison

    """## Part 1: linear regression with one variable: food truck
    <br>gradient_descent function at the top of the notebook. Parameters printed below.
    <br>Cost function and model with data plotted below.
    """

    X, Y = get_data('Data/food_truck_data.txt', ',')  # standard deviation and average are not needed in part1
    # for the problem.The data consists of two columns; the first column is the population of a city and the second column is the profit of a food truck in that city.A negative value for profit indicates a loss.

    # the model is h(theta)=theta[0] *x0 + theta[1]*x1 = theta.T*X
    initial_theta = np.array([0, 0])  # initial value for theta
    alpha = 0.002  # learning rate hyperparameter
    iterations = 10000  # number of iterations
    theta, cost = gradient_descent(X, Y, initial_theta, alpha, iterations)
    print(initial_theta)
    plot_cost(cost)
    plot_model(X[:, 1], Y, theta)
    print('example prediction: \npopulation(in 10,000s): 17.5 \nProfit(in $10,000): ', theta[0] + theta[1] * 17.5)

    """## Part 2: Linear regression with multiple variables: housing price
    <br>Features scaled in get_data function. Input is scaled and output is not.
    <br>gradient_descent function at the top of the notebook. Parameters printed below.
    <br>Cost function plotted below.
    <br>Sample prediction below.

    """

    # features (but not output) scaled in get_data
    X, Y, std_X, avg_X = get_data('Data/housing_price_data.txt', ',')

    # the model is h(theta)=theta[0] *x0 + theta[1]*x1 + theta[2]*x2 = theta.T*X
    initial_theta = np.array([0, 0, 0])  # initial value for theta
    alpha = 0.02  # learning rate hyperparameter
    iterations = 10000  # number of iterations
    theta, cost = gradient_descent(X, Y, initial_theta, alpha, iterations)

    print('theta = ', theta, '\nmodel: h(theta)= ', theta[0], ' + ', theta[1], ' * x1 + ', theta[2], '* x2',
          '\n(input scaled, output not scaled)')
    plot_cost(cost)

    """**Prediction:**
    Assume features of a house and predict a good market price: square feet, bedrooms
    """

    house = np.array([2000, 4])
    scaled_house = (house - avg_X) / std_X
    print('predicted price for house of ', house[0], 'square feet and ', house[1], 'bedrooms: $',
          theta[0] + theta[1] * scaled_house[0] + theta[2] * scaled_house[1])

if __name__ == '__main__':
    main()