import os
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects  as plot

import tkinter.ttk
from tkinter import ttk, filedialog, messagebox, Tk, StringVar
import customtkinter as tk
from PIL import Image
from tkcalendar import DateEntry

import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        Processbar['value'] += 1
        window.update_idletasks()
        print("\nEnd epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_batch_end(self, batch, logs=None):
        window.update()

class LSTMclass:
    def __init__(self,data,opt,los,epo,batch_si,nlayer,fig_stat,nunit):
        self.df = data
        self.optimizer = opt
        self.loss = los
        self.epoch = epo
        self.batch = batch_si
        self.layer = nlayer
        self.unit_p_layer = int(nunit)
        self.fig_state = fig_stat
        self.test_end = self.df.iloc[0]["Date"]
        self.test_start = self.df.iloc[60]["Date"]

    def run(self):
        # Add new column ['Tomorrow']
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

        # Input layer
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        # Hidden layer
        for i in range(0,self.layer):
            model.add(LSTM(units=self.unit_p_layer, return_sequences=True))
            model.add(Dropout(0.1))

        #output layer
        model.add(LSTM(units=10))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # dicided what optimizer and loss function will be use
        model.compile(optimizer=self.optimizer, loss=self.loss )

        # Begin training
        model.fit(x_train, y_train, epochs=self.epoch, batch_size=self.batch ,callbacks=[CustomCallback()])

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
        img_predict.configure(text="Loading...")
        window.update_idletasks()
        fig = plot.Figure([plot.Scatter(y=actual_prices_float,name = "Real_values"),plot.Scatter(y = to_list2d(prediction_prices),name = "Predict")])
        if self.fig_state == 1:
            fig.show()
        else:
            fig.write_image("fig2.jpeg")
        return prediction





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
    df["Price"] = df["Price"].str.replace(',', '').astype(float).fillna(0.0)
    df["Open"] = df["Open"].str.replace(',', '').astype(float).fillna(0.0)
    df["High"] = df["High"].str.replace(',', '').astype(float).fillna(0.0)
    df["Low"] = df["Low"].str.replace(',', '').astype(float).fillna(0.0)


    return df

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    stock_data.columns = ['Open', 'High', 'Low', 'Price', 'Vol']
    stock_data.reset_index(inplace=True)

    return stock_data

# Set appearance
tk.set_appearance_mode('dark')
tk.set_default_color_theme('blue')

# init window frame
window = tk.CTk()

df = pd.DataFrame
name = ''

# config window frame
window.minsize(width=1280,height=720)
window.title('Predict MaketPlace')

# cofig window grid
window.columnconfigure(0,weight=1)
window.columnconfigure(1,weight=2)

# frame input
frame_input = tk.CTkFrame(master=window)

# Token
label_name = tk.CTkLabel(master=frame_input,text="ID")
label_name.pack()

input_label_name = tk.CTkTextbox(master=frame_input,height=20)
input_label_name.pack()

# Date
label_cal_start = tk.CTkLabel(master=frame_input,text="Date start")
label_cal_start.pack()
cal_start = DateEntry(frame_input, width=12, year=2019, month=6, day=22,
background='darkblue', foreground='white', borderwidth=2)
cal_start.pack(padx=10, pady=10)

label_cal_end = tk.CTkLabel(master=frame_input,text="Date end")
label_cal_end.pack()
cal_end = DateEntry(frame_input, width=12, year=2022, month=1, day=1,
background='darkblue', foreground='white', borderwidth=2)
cal_end.pack(padx=10, pady=10)
output_resurt = tk.CTkLabel(master=frame_input,text='Watting...')

date_start = cal_start.get_date()
date_end = cal_end.get_date()

def get_date_from_cal():
    global date_start,date_end,df
    date_start = date_start.strftime("%Y-%m-%d")
    date_end = date_end.strftime("%Y-%m-%d")
    try:
        df = get_stock_data(input_label_name.get(1.0, "end-1c"),date_start,date_end)
        output_resurt.configure(text="Get data success.", text_color='#21ed28')
    except Exception as e:
        output_resurt.configure(text = ("Get data failure. " + str(e)),text_color='#e8132b' )
    show_table()


ik = tk.CTkButton(master=frame_input,text="Get Data", command= lambda :get_date_from_cal())
ik.pack()
output_resurt.pack()
frame_input.grid(row=1,column=0)


# frame data
frame_data = tk.CTkFrame(master=window,width= np.floor(window.winfo_screenwidth()*(2/5)),height=280)

# Show data_name choise
input_name = tk.CTkLabel(master=frame_data,height=20,width=200,text='')
input_name.pack()

# command for reading file csv
def file_input():
    global df,name
    name = filedialog.askopenfilename(title='Select csv file',filetypes=[('comma-separated values', '*.csv')])
    input_name.configure(text=name)

    # Try to open file
    try:
        df = pd.read_csv(name, date_parser = True).dropna()
    except Exception as e:
        messagebox.showerror("Something wrong ! ",str(e))
    show_table()

def show_table():
    global df
    # Get header
    table['column'] = list(df.columns)
    table['show'] = 'headings'
    # Show header
    for col in table['column']:
        table.heading(col,text=col)
        table.column(col,width=100,anchor=tk.CENTER, stretch=True)
    # Show data
    df_rows = df.to_numpy().tolist()
    for row in df_rows:
        table.insert("","end", values=row)

    # Editing data
    try:
        df = replace_comma(df)
    except Exception as e:
        print(e)
    df['Date'] = df['Date'].astype('datetime64[ns]')
    df = df.sort_values(by='Date', ascending=True)

# Read csv file button
input_button = tk.CTkButton(master=frame_data,text = 'Open file',
                            command=lambda :file_input(),
                            width=100,height=25,
                            corner_radius=5,
                            fg_color='#336fd6',hover_color='#0a1f42')
input_button.pack()


# Data Table for view data
table = ttk.Treeview(frame_data)

# vertical scrollbar
vsb = ttk.Scrollbar(frame_data, orient="vertical", command=table.yview)
vsb.pack(side='right', fill='y')

# Table style
table_style = ttk.Style()
table_style.theme_use('default')
# - for row below heading
table_style.configure("Treeview",
                      background = '#374a6b',
                      foreground= 'black',
                      fieldbackground='#374a6b')
# - for heading
table_style.configure("Treeview.Heading",
                      background='#162236',
                      foreground='black',
                      fieldbackground='#162236')
# Set scrollbar for table
table.configure(yscrollcommand=vsb.set)

table.pack(fill='both',expand=False)

# Config for table not interupt width of grid column
frame_data.pack_propagate(0)
frame_data.grid(row=1,column=1)

# Turn data to plot and save as image to show in frame
def show_input_plot_interactive(df):
    try:
        fig = plot.Figure(data=[plot.Candlestick(x=df['Date'],
                                                open=df['Open'],
                                                high=df['High'],
                                                low=df['Low'],
                                                close=df['Price'],
                                                 name = 'Candlestick'),


                                plot.Scatter(x = df['Date'],
                                          y = df['Price'],
                                          name = 'Price',
                                          visible='legendonly',
                                          line=dict(color='#3b24d1'))])
        fig.update_layout(xaxis_rangeslider_visible=False,title_text = "")
        fig.show()
    except Exception as e:
        print(str(e))

# Tab
tab_display = tk.CTkTabview(master=window)

# Config tab
tab_display.add('Plot')
tab_display.add('Predict')
# Set first tab display
tab_display.set('Plot')

# Inside tab 'Plot'

frame_display = tk.CTkFrame(master=tab_display.tab('Plot'))

# Prepare grid display
frame_display.columnconfigure(0,weight=1)
frame_display.columnconfigure(1,weight=1)

# Button to show plot
show_plot = tk.CTkButton(master=frame_display, text='Show plot in browser (interactive)', command=lambda :show_input_plot_interactive(df),
                            width=100,height=25,
                            corner_radius=5,
                            fg_color='#336fd6',
                            hover_color='#0a1f42')
show_plot.grid(row=0,column=0)

# Set first image to show
img = tk.CTkImage(light_image=Image.open('base.jpeg'),size=(700,500))
img_label = tk.CTkLabel(master=frame_display, text="", image=img)
img_label.configure(width=1280,height=480)
def show_input_plot_img(df):
    try:
        fig = plot.Figure(data=[plot.Candlestick(x=df['Date'],
                                                 open=df['Open'],
                                                 high=df['High'],
                                                 low=df['Low'],
                                                 close=df['Price']
                                                 )])
        fig.update_layout(xaxis_rangeslider_visible=False)

        fig.write_image('fig1.jpeg')
        global img
        img = tk.CTkImage(light_image=Image.open('fig1.jpeg'),size=(700, 500))
        img_label.configure(image=img)
        img_label.image = img
    except Exception as e:
        print(str(e))

show_plot_img = tk.CTkButton(master=frame_display, text='Show plot in this window (static image)', command=lambda :show_input_plot_img(df),
                                width=100,height=25,
                                corner_radius=5,
                                fg_color='#336fd6',
                                hover_color='#0a1f42')
show_plot_img.grid(row=0,column=1)

img_label.grid(row=1,column=0,columnspan=2)

frame_display.grid(row=2,sticky='EW',columnspan = 2)


# Inside tab predict
frame_predict = tk.CTkFrame(master=tab_display.tab('Predict'))
frame_predict.columnconfigure(1,weight=10)
# define variable
optimizer ='adam'
loss = 'mean_squared_error'
batch_size = 20
epochs = 20
layer = 2
unit = 20
fig_state = tk.IntVar()
var_batch = StringVar()
var_batch.set('Batch_size: '+str(batch_size))
var_epoch = StringVar()
var_epoch.set('Epochs: '+ str(epochs))
var_layer = StringVar()
var_layer.set('Number of hiden layer: '+ str(layer))
var_unit = StringVar()
var_unit.set('Number of unit for train: '+ str(unit))
predict_price = StringVar()
predict_price.set("Predict price : ")

# Label Optimizer
# Label
optimizer_name = tk.CTkLabel(master=frame_predict,text='Optimizer')
optimizer_name.grid(column = 0, row = 0)
# function to get choicen optimizer
def get_optimizer(choice):
    global optimizer
    optimizer = choice
# Option of optimizer
optimizer_menu = tk.CTkOptionMenu(frame_predict, values=['SGD','RMSprop','Adam',
                                                  'AdamW','Adadelta','Adagrad',
                                                  'Adamax','Adafactor','Nadam','Ftrl'],
                                         command=get_optimizer)
optimizer_menu.grid(column = 0, row = 1)

# Label Loss function
loss_name = tk.CTkLabel(master=frame_predict,text='Loss')
loss_name.grid(column = 0, row = 2)
def get_loss(choice):
    global loss
    loss = choice
# option of loss function
loss_menu = tk.CTkOptionMenu(frame_predict, values=['mean_squared_error','mean_absolute_error',
                                             'mean_absolute_percentage_error',
                                             'mean_squared_logarithmic_error'],
                                         command=get_loss)
loss_menu.grid(column = 0, row = 3)

# Batch size label
# Function to get batch size
def get_batch_size(value):
    global batch_size
    batch_size =int(np.round(value))
    var_batch.set('Batch_size '+str(batch_size))
# Label
batch_size_label = tk.CTkLabel(master=frame_predict,textvariable=var_batch)
batch_size_label.grid(column = 0, row = 4)
# Silder
batch_size_slider = tk.CTkSlider(frame_predict, from_=20, to=200, command=get_batch_size ,width=500)
batch_size_slider.grid(column = 0, row = 5)

# Epochs Label
def get_epochs(value):
    global epochs
    epochs =int(np.round(value))
    var_epoch.set('Epochs: '+str(epochs))

epochs_label = tk.CTkLabel(master=frame_predict,textvariable=var_epoch)
epochs_label.grid(column = 0, row = 6)
epochs_slider = tk.CTkSlider(frame_predict, from_=1, to=100, command=get_epochs ,width=500)
epochs_slider.grid(column = 0, row = 7)
# Layer config

def get_layer(value):
    global layer
    layer =int(np.round(value))
    var_layer.set('Number of hiden layer: '+ str(layer))

layer_label = tk.CTkLabel(master=frame_predict,textvariable=var_layer)
layer_label.grid(column = 0, row = 8)
layer_slider = tk.CTkSlider(frame_predict, from_=1, to=15, command=get_layer ,width=500)
layer_slider.grid(column = 0, row = 9)
# Unit config
def get_unit(value):
    global unit
    unit =int(np.round(value))
    var_unit.set('Number of unit for train: '+ str(unit))

unit_label = tk.CTkLabel(master=frame_predict,textvariable=var_unit)
unit_label.grid(column = 0, row = 10)

unit_textbox = tk.CTkSlider(frame_predict, from_=1, to=200, command=get_unit ,width=500)
unit_textbox.grid(column = 0, row = 11)

# Show plot
checkbox_plot = tk.CTkCheckBox(master=frame_predict, variable=fig_state, onvalue=1,offvalue=0, text="Show interative figure")
checkbox_plot.grid(column = 0, row = 12)
def run_predict():
    global optimizer, loss, epochs, batch_size, img_predict, fig_state,layer ,unit
    Processbar.configure(maximum = epochs)
    Processbar['value'] = 0
    cur_run = LSTMclass(df, optimizer, loss, epochs, batch_size, layer, fig_state.get(),unit)

    values = cur_run.run()
    predict_price.set("Predict values : " + str(values))
    
    # temp.run(optimizer, loss, epochs, batch_size)
    if fig_state.get() == 0:
        img = tk.CTkImage(light_image=Image.open('fig2.jpeg'), size=(700, 500))
        img_predict.configure(image=img)
        img_predict.image = img

begin = tk.CTkButton(master=frame_predict,text='Run',command=lambda : run_predict())
begin.grid(column = 0,row=13)
Processbar = tkinter.ttk.Progressbar(master=frame_predict ,orient = tkinter.HORIZONTAL,length = 300, mode = 'determinate' ,maximum = epochs)
Processbar.grid(column = 0, row=14)

Predict_price = tk.CTkLabel(master=frame_predict,textvariable=predict_price)
Predict_price.grid(column =0 , row = 15)

img_pred = tk.CTkImage(light_image=Image.open('base.jpeg'),size=(700,500))
img_predict = tk.CTkLabel(master=frame_predict,image=img_pred, text="Notthing to show")
img_predict.grid(column = 1, row = 0, rowspan = 26)
frame_predict.pack()
tab_display.grid(row=2,sticky='EW',columnspan = 2)

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        path_1 = os.getcwd() + '\\fig1.jpeg'
        path_2 = os.getcwd() + '\\fig2.jpeg'

        if os.path.exists(path_1):
            os.remove(path_1)
        if os.path.exists(path_2):
            os.remove(path_2)
        window.destroy()

# Closing window
window.protocol("WM_DELETE_WINDOW", on_closing)
window.mainloop()

