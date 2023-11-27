import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import plotly.express as pl
import numpy as np


def to_list2d(df):
    d = []
    for i in df:
        d.append(i[0])
    return d


class LSTMclass:
    def __init__(self,data,optimizer,loss,epoch,batch_size,layer,fig_state):
        self.df = data
        self.optimizer = optimizer
        self.loss = loss
        self.epoch = epoch
        self.batch = batch_size
        self.layer = layer
        self.fig_state : bool = fig_state
        self.test_end = self.df.iloc[0]["Date"]
        self.test_start = self.df.iloc[60]["Date"]

    def run(self):
        # Add new column ['Tomorrow'] - Note: Make no impact
        self.df["Tomorrow"] = self.df["Price"].shift(-1)

        # prep data for training
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.df['Price'].values.reshape(-1, 1))

        # Set how many data to test-train (will recalculate)
        prediction_days = 100

        # training data
        x_train = []  # Input data
        y_train = []  # Target data

        # Set which data will be used for training
        for x in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x - prediction_days:x, 0])
            y_train.append(scaled_data[x, 0])

        # Finish prepare data
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # build model
        model = Sequential()

        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # dicided what optimizer and loss function will be use
        model.compile(optimizer=self.optimizer, loss=self.loss)

        # Begin training
        model.fit(x_train, y_train, epochs=self.epoch, batch_size=self.batch)

        # Prepare test data
        test_data = self.df.loc[(self.df['Date'] > self.test_start)]
        actual_prices = test_data["Price"]
        total_dataset = pd.concat((self.df["Price"], test_data["Price"]), axis=0)

        model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = scaler.transform(model_inputs)

        x_test = []

        for x in range(prediction_days, len(model_inputs)):
            x_test.append(model_inputs[x - prediction_days:x, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Testing
        prediction_prices = model.predict(x_test)
        prediction_prices = scaler.inverse_transform(prediction_prices)

        # Create data for predict

        '''
        next_data = [model_inputs[len(model_inputs)+ 1 - prediction_days:len(model_inputs + 1), 0]]
        '''

        next_data = [model_inputs[len(model_inputs) - prediction_days:, 0]]
        next_data = np.array(next_data)
        next_data = np.reshape(next_data, (next_data.shape[0], next_data.shape[1], 1))

        # Predict
        prediction = model.predict(next_data)
        prediction = scaler.inverse_transform(prediction)

        # Save date
        print(f"Prediction tomorow {prediction}")
        actual_prices_float = []
        for i in actual_prices.values:
            actual_prices_float.append(float(i))

        data_plot = [to_list2d(prediction_prices), actual_prices_float]
        fig = pl.line(y=data_plot, labels={"prediction_prices", "Actual"})
        if self.fig_state:
            fig.show()
        else:
            fig.write_image("fig2.png")


