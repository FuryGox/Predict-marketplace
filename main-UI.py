from tkinter import ttk, filedialog, messagebox,Tk
import customtkinter as tk
import pandas as pd
import investiny as inv
import plotly.graph_objects  as plot
import numpy as np
from PIL import Image
from tkcalendar import DateEntry

# Set appearance


tk.set_appearance_mode('dark')
tk.set_default_color_theme('blue')

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

label_name = tk.CTkLabel(master=frame_input,text="ID")
label_name.pack()

input_label_name = tk.CTkTextbox(master=frame_input,height=20)
input_label_name.pack()


cal_start = DateEntry(frame_input, width=12, year=2019, month=6, day=22,
background='darkblue', foreground='white', borderwidth=2)
cal_start.pack(padx=10, pady=10)

cal_end = DateEntry(frame_input, width=12, year=2022, month=1, day=1,
background='darkblue', foreground='white', borderwidth=2)
cal_end.pack(padx=10, pady=10)
output_resurt = tk.CTkLabel(master=frame_input,text='Watting..')


date_start = cal_start.get_date()
date_end = cal_end.get_date()
def get_date_from_cal():
    global date_start,date_end,df
    date_start = date_start.strftime("%m/%d/%Y")
    date_end = date_end.strftime("%m/%d/%Y")
    try:
        df = inv.historical_data(investing_id=input_label_name.get(1.0, "end-1c"),from_date=date_start,to_date=date_end)
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
    except Exception:
        print(Exception)
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
                                                close=df['Price']),

                                plot.Line(x = df['Date'],
                                          y = df['Price'],
                                          line=dict(color='#41cdf0'))])
        fig.update_layout(xaxis_rangeslider_visible=False,title_text = "")
        fig.show()
    except Exception as e:
        print(str(e))
    """
    figure = plt.Figure(figsize=(10, 4), dpi=100)
    st = figure.suptitle("suptitle", fontsize="x-large")
    df['Date'] = pd.to_datetime(df['Date'])

    # Plot for ['Open'] values
    ax1 = figure.add_subplot(141)
    ax1.plot(df['Date'],df['Open'].tolist(), color = '#1bd152')
    ax1.set_title('ax1')

    # Plot for ['Price'] values
    ax2 = figure.add_subplot(142)
    ax2.plot(df['Date'],df['Price'].tolist(),color= '#1f13f2')
    ax2.set_title('ax2')

    # Plot for ['High'] values
    ax3 = figure.add_subplot(143)
    ax3.plot(df['Date'],df['High'].tolist(),color = '#db8b2a')
    ax3.set_title('ax3')

    # Plot for ['Low'] values
    ax4 = figure.add_subplot(144)
    ax4.plot(df['Date'],df['Low'].tolist(), color = '#222fe3')
    ax4.set_title('ax4')

    st.set_y(0.95)
    figure.tight_layout()
    figure.subplots_adjust(top=0.85)
    scatter3 = FigureCanvasTkAgg(figure, frame_display)
    scatter3.draw()
    scatter3.get_tk_widget().pack(fill='x')
    toolbar = NavigationToolbar2Tk(scatter3,frame_display)
    toolbar.update()
    scatter3.get_tk_widget().pack()
    """


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

        fig.write_image('fig.jpeg')
        global img
        img = tk.CTkImage(light_image=Image.open('fig.jpeg'),size=(700, 500))
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



frame_predict = tk.CTkFrame(master=tab_display.tab('Predict'))

optimizer ='adam'
loss = 'mean_squared_error'
batch_size = 20
epochs = 20

optimizer_name = tk.CTkLabel(master=frame_predict,text='Optimizer')
optimizer_name.pack()
def get_optimizer(choice):
    global optimizer
    optimizer = choice

optimizer_menu = tk.CTkOptionMenu(frame_predict, values=['SGD','RMSprop','Adam',
                                                  'AdamW','Adadelta','Adagrad',
                                                  'Adamax','Adafactor','Nadam','Ftrl'],
                                         command=get_optimizer)
optimizer_menu.pack()

loss_name = tk.CTkLabel(master=frame_predict,text='Loss')
loss_name.pack()
def get_loss(choice):
    global loss
    loss = choice

loss_menu = tk.CTkOptionMenu(frame_predict, values=['mean_squared_error','mean_absolute_error',
                                             'mean_absolute_percentage_error',
                                             'mean_squared_logarithmic_error'],
                                         command=get_loss)
loss_menu.pack()
def get_batch_size(value):
    global batch_size
    batch_size = np.round(value)

batch_size_label = tk.CTkLabel(master=frame_predict,text='batch_size')
batch_size_label.pack()
batch_size_slider = tk.CTkSlider(frame_predict, from_=20, to=100, command=get_batch_size)
batch_size_slider.pack()
def get_epochs(value):
    global epochs
    epochs = np.round(value)

epochs_label = tk.CTkLabel(master=frame_predict,text='epochs')
epochs_label.pack()
epochs_slider = tk.CTkSlider(frame_predict, from_=20, to=100, command=get_epochs)
epochs_slider.pack()
def out():
    global loss,optimizer,batch_size,epochs
    print(loss,optimizer,batch_size,epochs)

t = tk.CTkButton(master=frame_predict,command=lambda :out(),text="hitme")
t.pack()
frame_predict.grid(row=2,sticky='EW',columnspan = 2)
tab_display.grid(row=2,sticky='EW',columnspan = 2)
window.mainloop()