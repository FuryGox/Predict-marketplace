from tkinter import *
from tkinter import ttk
import numpy as np
import customtkinter as tk
window = tk.CTk()

optimizer ='adam'
loss = 'mean_squared_error'
batch_size = 20
epochs = 20

optimizer_name = tk.CTkLabel(master=window,text='Optimizer')
optimizer_name.pack()
def get_optimizer(choice):
    global optimizer
    optimizer = choice

optimizer_menu = tk.CTkOptionMenu(window, values=['SGD','RMSprop','Adam',
                                                  'AdamW','Adadelta','Adagrad',
                                                  'Adamax','Adafactor','Nadam','Ftrl'],
                                         command=get_optimizer)
optimizer_menu.pack()

loss_name = tk.CTkLabel(master=window,text='Loss')
loss_name.pack()
def get_loss(choice):
    global loss
    loss = choice

loss_menu = tk.CTkOptionMenu(window, values=['mean_squared_error','mean_absolute_error',
                                             'mean_absolute_percentage_error',
                                             'mean_squared_logarithmic_error'],
                                         command=get_loss)
loss_menu.pack()
def get_batch_size(value):
    global batch_size
    batch_size = np.round(value)

batch_size_label = tk.CTkLabel(master=window,text='batch_size')
batch_size_label.pack()
batch_size_slider = tk.CTkSlider(window, from_=20, to=100, command=get_batch_size)
batch_size_slider.pack()
def get_epochs(value):
    global epochs
    epochs = np.round(value)

epochs_label = tk.CTkLabel(master=window,text='epochs')
epochs_label.pack()
epochs_slider = tk.CTkSlider(window, from_=20, to=100, command=get_epochs)
epochs_slider.pack()
def out():
    global loss,optimizer,batch_size,epochs
    print(loss,optimizer,batch_size,epochs)

t = tk.CTkButton(master=window,command=lambda :out(),text="hitme")
t.pack()
window.mainloop()