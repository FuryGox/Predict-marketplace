import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
df = pd.read_csv('Bitcoin Historical Data.csv')

features = ['Open', 'High', 'Low', 'Vol.']
target = 'Price'


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


df = df.dropna()

try:
    replace_comma(df)
except Exception:
    print(Exception)
print(df)

# Chuyển dữ liệu thành mảng numpy và chuẩn hóa dữ liệu
features = ['Open', 'High', 'Low', 'Vol.']
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
    for i in range(len(dataset)-look_back):
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
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Biên soạn và huấn luyện mô hình
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Dự đoán giá trong tương lai
predictions = []

for i in range(len(X_test)):
    x_input = X_test[i].reshape((1, look_back, X_test.shape[2]))
    y_pred = model.predict(x_input, verbose=0)
    predictions.append(y_pred[0, 0])

# Chuyển ngược quy trình chuẩn hóa để lấy giá trị dự đoán thực tế
predicted_prices = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))

# Vẽ biểu đồ giá thực tế và dự đoán
plt.plot(y[-len(test):], label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
"""
# Tính toán giá trị trung bình của giá cổ phiếu theo ngày
df['Price_Mean'] = df['Price'].rolling(window=5).mean()

# Loại bỏ các dòng chứa giá trị NaN (do phương thức rolling)
df.dropna(inplace=True)

# Chuyển dữ liệu thành DataFrame của pandas và đặt cột 'Date' làm index
df = pd.DataFrame(df)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Tính toán giá trị trung bình của giá cổ phiếu theo ngày
df['Price_Mean'] = df['Price'].rolling(window=5).mean()

# Loại bỏ các dòng chứa giá trị NaN (do phương thức rolling)
df.dropna(inplace=True)

# Chuyển dữ liệu thành mảng numpy và chuẩn hóa dữ liệu
features = ['Open', 'High', 'Low', 'Vol.', 'Price_Mean']
X = np.array(df[features])
y = np.array(df['Price'])
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Reshape dữ liệu để phù hợp với đầu vào của mô hình LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Biên soạn và huấn luyện mô hình
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Đánh giá mô hình trên tập kiểm tra
score = model.evaluate(X_test, y_test, batch_size=16)
print("MSE on test set:", score)

# Dự đoán giá trị mới
new_data = np.array([[...]])  # Thay thế ... bằng giá trị thí nghiệm của bạn
new_data_scaled = scaler_X.transform(new_data)
new_data_reshaped = new_data_scaled.reshape((1, 1, new_data_scaled.shape[1]))
predicted_price_scaled = model.predict(new_data_reshaped)
predicted_price = scaler_y.inverse_transform(predicted_price_scaled)
print("Predicted Price:", predicted_price)

# Visualize the predictions
y_pred = model.predict(X_test)
y_pred_original = scaler_y.inverse_transform(y_pred)
y_test_original = scaler_y.inverse_transform(y_test)

plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(y_test):], y_test_original, label='True Values')
plt.plot(df.index[-len(y_test):], y_pred_original, label='Predictions')
plt.title('LSTM Model Predictions vs True Values')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()"""