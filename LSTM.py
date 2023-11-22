import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df = pd.read_csv('Bitcoin Historical Data.csv',parse_dates=True).dropna()
#df = df.set_index('Date')


# Covert mark M ,K ,B
def convert_mark(x):
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
def to_list2d(data):
    d = []
    for i in data:
        d.append(i[0])
    return d

def replace_comma(df):
    vol = []
    for i in df["Vol."]:
        vol.append(convert_mark(i))
    df["Vol."] = vol
    df["Price"] = df["Price"].str.replace(',', '').astype(float)
    df["Open"] = df["Open"].str.replace(',', '').astype(float)
    df["High"] = df["High"].str.replace(',', '').astype(float)
    df["Low"] = df["Low"].str.replace(',', '').astype(float)


    return df


try:
    df = replace_comma(df)
except Exception as e:
    print(e)
df['Date'] = df['Date'].astype('datetime64[ns]')
features = ['Open', 'High', 'Low']
X = np.array(df[features])
y = np.array(df['Price'])
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_size = int(len(X_scaled) * 0.8)
train, test = X_scaled[0:train_size], X_scaled[train_size:len(X_scaled)]

# Chia dữ liệu thành đầu vào (X) và đầu ra (y)
def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        X.append(a)
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 10  # Số bước thời gian để dự đoán bước thời gian tiếp theo
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# Reshape dữ liệu để phù hợp với đầu vào của mô hình LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(50))
model.add(LSTM(units=30))
model.add(Dropout(0.2))
model.add(LSTM(units=10))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Biên soạn và huấn luyện mô hình
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Dự đoán giá ngày tiếp theo
last_sequence = X_scaled[-look_back:]
last_sequence = last_sequence.reshape((1, look_back, len(features)))
next_day_price_scaled = model.predict(last_sequence)
next_day_price = scaler_y.inverse_transform(next_day_price_scaled.reshape(-1, 1))[0, 0]
print("Predicted Price for the Next Day:", next_day_price)