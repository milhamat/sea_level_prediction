import numpy as np
import matplotlib.pyplot as plt

class ShowPlot:
    def load_data(self, data_path: str="sea-level.csv") -> float:
        """this function is for data reading and simple data preprocessing.

        Args:
            data_path (str, optional): put file path and your dataset name. 
            Defaults to "sea-level.csv".

        Returns:
            float: it will returns as your dataset file. 
        """
        # load the data as numpy
        data = np.genfromtxt(data_path, delimiter=",")
        # the second column of the data is useless
        data = np.delete(data, 1, 1)
        # there are some NaN rows, remove them
        data = data[~np.isnan(data).any(axis=1)]
        return data

    def create_plot(self, model, x: float, y: float, year: int) -> None:
        """This function helps to execute your model and bring the model 
           results in form of graphical diagram.

        Args:
            model (_type_): your model either linear model or quardatic model.
            x (float): the independent variable
            y (float): the dependent variable 
            year (int): your prediction year
        """
        # Make model prediction for historical data
        y_pred = model.predict(x)
        # We can calculate the error (MAE) since we have the actual y
        mae = np.mean(np.abs(y - y_pred))

        # The last year that we have in the data
        last_year = x.max()
        # Create year sequences, from the last year in data until target forecast year
        x_forecast = np.arange(last_year, year + 1)
        y_forecast_pred = model.predict(x_forecast)

        # Write some results text for plot title
        title = "Sea Level Prediction\n"
        title += f"Mean Asolute Error: {mae:.2f} mm\n"
        title += f"Sea Level {int(x_forecast[-1])}: {y_forecast_pred[-1]:.2f} mm"

        # Initialize the plot
        _, ax = plt.subplots(figsize=(8, 8))
        # Draw actual data
        ax.scatter(x, y, label="Actual Data")
        # Draw historical prediction
        ax.plot(x, y_pred, color="red", label="Prediction (Past)")
        # Draw future prediction
        ax.plot(x_forecast, y_forecast_pred, "--", color="red", label="Prediction (Future)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Sea Level (mm)")
        ax.set_title(title)
        ax.legend()
        plt.show()
    