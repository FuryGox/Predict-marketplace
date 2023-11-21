import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
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

df.set_index('Date', inplace=True)

features = ['Open', 'High', 'Low', 'Vol.']
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