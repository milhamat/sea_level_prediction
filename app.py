from src import LinearModel, QuadraticModel, ShowPlot

def main():
    plot = ShowPlot()
    # data = plot.load_data(plot.data_name)
    data = plot.load_data()
    x_data, y_data = data[:, 0], data[:, 1]

    forecast_year = int(input("What year do you want to perform forecast? "))
    
    if forecast_year < x_data[-1]:
        raise ValueError(f"The forecast year must be > {x_data[-1]}")

    model_type = int(input("What model degree you want to use (1: linear, 2: quadratic)? "))

    if model_type == 1:
        print("Using linear model")
        model = LinearModel()
        model.train(x_data, y_data)
        plot.create_plot(model, x_data, y_data, forecast_year)

    elif model_type == 2:
        print("Using quadratic model")
        model = QuadraticModel()
        model.train(x_data, y_data)
        plot.create_plot(model, x_data, y_data, forecast_year)


if __name__ == "__main__":
    main()
