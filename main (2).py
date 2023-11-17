from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

name = 'Bitcoin Historical Data.csv'

def convert_M(x):
    if x.endswith("M"):
        x = x.replace("M",' ')
        x = float(x) * 1000000
        return x
    elif x.endswith("K"):
        x = x.replace("K",' ')
        x = float(x) * 1000
        return x
    elif x.endswith("B"):
        x = x.replace("B",' ')
        x = float(x) * 1000000000
        return x
    else:
        return x
def to_list2D(data):
    d = []
    for i in data:
        d.append(i[0])
    return d

def replace_comma(df):
    df["Price"] = df["Price"].str.replace(',', '')
    df["Open"] = df["Open"].str.replace(',', '')
    df["High"] = df["High"].str.replace(',', '')
    df["Low"] = df["Low"].str.replace(',', '')
    vol = []
    for i in df["Vol."]:
        vol.append(convert_M(i))
    df["Vol."] = vol


df = pd.read_csv(name)
df = df.dropna()
try:
    replace_comma(df)
except Exception:
    print(Exception)
print(df)


test_end = df.iloc[0]["Date"]
test_start = df.iloc[60]["Date"]

#prep data for training
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df['Price'].values.reshape(-1,1))

#training data
x_train = []        #Input data
y_train = []        #Target data

#Set how many data to test-train (will recalculate)
prediction_days = 60

#Set which data will be used for training
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x,0])
    y_train.append(scaled_data[x,0])

#Finish prepare data
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))

model = Sequential()

model.add(LSTM(units=100, return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=75, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))


#dicided what optimizer and loss function will be use
model.compile(optimizer="adam", loss='mean_squared_error')

#Begin training
model.fit(x_train,y_train,epochs=35,batch_size=64)

# Prepare test data
test_data = df.loc[(df['Date'] > test_start)]
actual_prices = test_data["Price"]
total_dataset = pd.concat((df["Price"], test_data["Price"]), axis=0)

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

#Create data for predict

'''
next_data = [model_inputs[len(model_inputs)+ 1 - prediction_days:len(model_inputs + 1), 0]]
'''

next_data = [model_inputs[len(model_inputs) - prediction_days:, 0]]
next_data = np.array(next_data)
next_data = np.reshape(next_data, (next_data.shape[0], next_data.shape[1],1))

#Predict
prediction = model.predict(next_data)
prediction = scaler.inverse_transform(prediction)

#Save date
global a
a = prediction
print(f"Prediction tomorow {prediction}")
actual_prices_float = []
for i in actual_prices.values:
    actual_prices_float.append(float(i))


fig = plt.plot(to_list2D(prediction_prices))
plt.plot(actual_prices_float)
plt.show()
