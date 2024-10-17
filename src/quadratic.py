import numpy as np

class QuadraticModel:
    def __init__(self):
        # initiate random constants a, b, and c
        self.weights = np.random.rand(3)
        self.x_mean = 0
        self.x_std = 1
        self.y_mean = 0
        self.y_std = 1

    def predict(self, x: float) -> float:
        """this function is for calculating the predicting data 

        Args:
            x (float): the independent variable

        Returns:
            float: y-pred de-normalized
        """
        # the input x should normalized first,
        x_norm = (x - self.x_mean) / self.x_std
        # do the prediction
        y_pred_norm = self.calculate_y(x_norm)
        # and the output y should be de-normalized
        return y_pred_norm * self.y_std + self.y_mean

    def calculate_y(self, x: float) -> float:
        """Calculate the matematical formulation of Quadratic model

        Args:
            x (float): the independent variable

        Returns:
            float: the weight value from independent variable
        """
        # Calculate the y = ax^2 + bx + c
        return self.weights[0] * x**2 + self.weights[1] * x + self.weights[2]

    def calculate_gradient(self, x: float, y: float, y_pred: float) -> float:
        """for calculating the gradient 

        Args:
            x (float): the independent variable
            y (float): the dependent variable
            y_pred (float): the new data result 

        Returns:
            float: gradient weight
        """
        # Calculate the gradient
        # dL/da = 2 * x^2 * (pred - actual)
        # dL/db = 2 * x * (pred - actual)
        # dL/dc = 2 * (pred - actual)
        d_weights = self.weights.copy()
        d_weights[0] = np.mean(2 * (y_pred - y) * x**2)
        d_weights[1] = np.mean(2 * (y_pred - y) * x)
        d_weights[2] = np.mean(2 * (y_pred - y))
        return d_weights

    def gradient_descend(self, x: float, y: float, learning_rate: float) -> None:
        """to calculating the gradient descend

        Args:
            x (float): the independent variable
            y (float): the dependent variable
            learning_rate (float): hyperparameter that 
            determines how quickly a model converges to an optimal solution
        """
        # First, normalize all input/output so that we get better training
        # normalized_value = (original_value - mean) / standard_deviation
        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        # Do the gradient descent
        y_pred = self.calculate_y(x)
        d_weights = self.calculate_gradient(x, y, y_pred)
        self.weights -= learning_rate * d_weights

    def train(self, x_data: float, y_data: float, max_epochs: int=50, learning_rate: float=0.1) -> None:
        """to training the model

        Args:
            x_data (float): the independent variable
            y_data (float): the dependent variable
            max_epochs (int, optional): maximum number of training cycle . Defaults to 50.
            learning_rate (float, optional): hyperparameter that 
            determines how quickly a model converges to an optimal solution. Defaults to 0.1.
        """
        print("Start training")

        # First let's calculate the mean and standard deviation for each
        # input x and output y. Those information will be usefil for
        # data normalization.
        self.x_mean = np.mean(x_data)
        self.x_std = np.std(x_data)
        self.y_mean = np.mean(y_data)
        self.y_std = np.std(y_data)
        # Then, perform gradient descent for as much as the number of epochs
        for _ in range(max_epochs):
            self.gradient_descend(x_data, y_data, learning_rate)

        print("Finished training")